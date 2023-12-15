import { dispatchProgress, loadModel, PretrainedOptions, ProgressCallback, ProgressStatus } from '@/pipelines/common'
import { Session } from '@/backends'
import { LCMScheduler, LCMSchedulerConfig } from '@/schedulers/LCMScheduler'
import { GetModelFileOptions } from '@/hub/common'
import { CLIPTokenizer } from '@/tokenizers/CLIPTokenizer'
import { getModelJSON } from '@/hub'
import { cat, linspace, randomNormalTensor, range } from '@/util/Tensor'
import { Tensor } from '@xenova/transformers'
import { DiffusionPipeline } from '@/pipelines/DiffusionPipeline'
import { PipelineBase } from '@/pipelines/PipelineBase'
import * as tf from '@tensorflow/tfjs';

export interface SimilarImageInput {
  prompt: string
  numImages: number
  differentiation: number
  negativePrompt?: string
  guidanceScale?: number
  seed?: string
  width?: number
  height?: number
  numInferenceSteps: number
  sdV1?: boolean
  progressCallback?: ProgressCallback
  runVaeOnEachStep?: boolean
  img2imgFlag?: boolean
  inputImage?: Float32Array
  strength?: number
}

export class SimilarImagePipeline extends PipelineBase {
  declare public scheduler: LCMScheduler

  constructor (unet: Session, vaeDecoder: Session, vaeEncoder: Session, textEncoder: Session, tokenizer: CLIPTokenizer, scheduler: LCMScheduler) {
    super()
    this.unet = unet
    this.vaeDecoder = vaeDecoder
    this.vaeEncoder = vaeEncoder
    this.textEncoder = textEncoder
    this.tokenizer = tokenizer
    this.scheduler = scheduler
    this.vaeScaleFactor = 2 ** ((this.vaeDecoder.config.block_out_channels as string[]).length - 1)
  }

  static createScheduler (config: LCMSchedulerConfig) {
    return new LCMScheduler(
      {
        prediction_type: 'epsilon',
        ...config,
      },
    )
  }

  static async fromPretrained (modelRepoOrPath: string, options?: PretrainedOptions) {
    const opts: GetModelFileOptions = {
      ...options,
    }

    // order matters because WASM memory cannot be decreased. so we load the biggest one first
    const unet = await loadModel(
      modelRepoOrPath,
      'unet/model.onnx',
      opts,
    )
    const textEncoder = await loadModel(modelRepoOrPath, 'text_encoder/model.onnx', opts)
    const vaeEncoder = await loadModel(modelRepoOrPath, 'vae_encoder/model.onnx', opts)
    const vae = await loadModel(modelRepoOrPath, 'vae_decoder/model.onnx', opts)
    const schedulerConfig = await getModelJSON(modelRepoOrPath, 'scheduler/scheduler_config.json', true, opts)
    const scheduler = SimilarImagePipeline.createScheduler(schedulerConfig)

    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer' })
    await dispatchProgress(opts.progressCallback, {
      status: ProgressStatus.Ready,
    })
    return new SimilarImagePipeline(unet, vae, vaeEncoder, textEncoder, tokenizer, scheduler)
  }

  getWEmbedding (batchSize: number, guidanceScale: number, embeddingDim = 512) {
    let w = new Tensor('float32', new Float32Array([guidanceScale]), [1])
    w = w.mul(1000)

    const halfDim = embeddingDim / 2
    let log = Math.log(10000) / (halfDim - 1)
    let emb: Tensor = range(0, halfDim).mul(-log).exp()

    // TODO: support batch size > 1
    emb = emb.mul(w.data[0])

    return cat([emb.sin(), emb.cos()]).reshape([batchSize, embeddingDim])
  }

  async release () {
    await this.unet?.release()
    await this.vaeDecoder?.release()
    await this.vaeEncoder?.release()
    await this.textEncoder?.release()
  }

  async run (input: SimilarImageInput) {
    const width = input.width || this.unet.config.sample_size as number * this.vaeScaleFactor
    const height = input.height || this.unet.config.sample_size as number * this.vaeScaleFactor
    const batchSize = 1
    const guidanceScale = input.guidanceScale || 1
    const seed = input.seed || '69'
    this.scheduler.setTimesteps(input.numInferenceSteps || 5)

    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.EncodingPrompt,
    })

    const promptEmbeds = await this.encodePrompt(input.prompt)

    const latents_x = tf.randomNormal([4, 64, 64], 0, 1, 'float32', parseInt(seed))      // mean: 0, std: 1 as per codebase randomNormalTensor
    const latents_y = tf.randomNormal([4, 64, 64], 0, 1, 'float32', parseInt(seed)*180)

    const scale_x = tf.cos(tf.linspace(0, 1, input.numImages).mul(Math.PI).mul(2).mul(input.differentiation))
    const scale_y = tf.sin(tf.linspace(0, 1, input.numImages).mul(Math.PI).mul(2).mul(input.differentiation))
    
    const noise_x = [];                              // eq of tensordot
    for (let i=0; i<scale_x.size; i++) {
      let x = tf.mul(latents_x, scale_x.slice([i],[1]));
      noise_x.push(x);
    }

    const noise_y = [];                              // eq of tensordot
    for (let i=0; i<scale_y.size; i++) {
      let y = tf.mul(latents_y, scale_y.slice([i],[1]));
      noise_y.push(y);
    }
    
    const noiseX = tf.stack(noise_x);               // 10 x 4 x 64 x 64
    const noiseY = tf.stack(noise_y);               // 10 x 4 x 64 x 64
    
    const noise = tf.add(noiseX, noiseY)
    const batched_noise = tf.split(noise, input.numImages)       // 1 x 4 x 64 x 64
    console.log(batched_noise[0].shape)

    async function processTimesteps (unet: Session, scheduler: LCMScheduler, latents: Tensor, promptEmbeds: Tensor) {
      let timesteps = scheduler.timesteps.data
      let humanStep = 1
      let denoised: Tensor
      for (const step of timesteps) {
        const timestep = new Tensor(new Float32Array([step]))
        await dispatchProgress(input.progressCallback, {
          status: ProgressStatus.RunningUnet,
          unetTimestep: humanStep,
          unetTotalSteps: timesteps.length,
        })
    
        const noise = await unet.run(
          { sample: latents, timestep, encoder_hidden_states: promptEmbeds},
        );
        [latents, denoised] = scheduler.step(
          noise.out_sample,
          step,
          latents,
        )
        humanStep++
      }

      await dispatchProgress(input.progressCallback, {
        status: ProgressStatus.Done,
      })
      return denoised;
    }

    const images = []
    for (let i=0; i<input.numImages; i++) {
      this.scheduler.setTimesteps(input.numInferenceSteps || 5)
      let batchJS = batched_noise[i].dataSync()                          // tf.Tensor -> Float32Array
      let batchTensor = new Tensor('float32', batchJS, [1, 4, 64, 64])   // Float32Array -> Tensor
      let denoised = await processTimesteps(this.unet, this.scheduler, batchTensor, promptEmbeds)
      images.push(await this.makeImages(denoised))
    }

    return [images];
  }
}
