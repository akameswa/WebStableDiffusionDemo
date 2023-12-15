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

interface StableDiffusionInput {
  prompt: string
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

export class LatentConsistencyModelPipeline extends PipelineBase {
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
    const scheduler = LatentConsistencyModelPipeline.createScheduler(schedulerConfig)

    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer' })
    await dispatchProgress(opts.progressCallback, {
      status: ProgressStatus.Ready,
    })
    return new LatentConsistencyModelPipeline(unet, vae, vaeEncoder, textEncoder, tokenizer, scheduler)
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

  async run (input: StableDiffusionInput) {
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

    // NOTE: latentsMod is same as latents despite the loop (check inference at 1). hence, swifted to tf
    // let latents = this.prepareLatents(
    //   batchSize,
    //   this.unet.config.in_channels as number || 4,
    //   height,
    //   width,
    //   seed,
    // )

    // let pokeLatents = this.prepareLatents(
    //   batchSize,
    //   this.unet.config.in_channels as number || 4,
    //   height/4,         // 128
    //   width/4,
    //   seed + '69420',
    // )

    // let latentsMod = latents
    // for (let i = 0; i < height/(4*this.vaeScaleFactor); i++) {
    //   for (let j = 0; j < width/(4*this.vaeScaleFactor); j++) {
    //       for (let k = 0; k < 4; k++) {
    //           latentsMod[0][k][i][j] = pokeLatents[0][k][i][j];
    //       }
    //   }
    // }

    const latents = tf.randomNormal([4, 64, 64], 0, 1, 'float32', parseInt(seed))      // mean: 0, std: 1 as per codebase randomNormalTensor
    const newlatents = tf.randomNormal([4, 16, 16], 0, 1, 'float32', parseInt(seed)*180)

    const left = latents.slice([0, 0, 0], [4, 64, 16])                                 // height 64, width 16
    const right = latents.slice([0, 0, 16], [4, 64, 48])                               // cut at width 16, height 64, remaining width 48
    const top_left = left.slice([0, 0, 0], [4, 16, 16])
    const bottom_left = left.slice([0, 16, 0], [4, 48, 16])
    const leftMod = tf.concat([newlatents, bottom_left],1)
    const latentsMod = tf.concat([leftMod, right],2)

    const jsLatents = latents.dataSync()                                                 // tf.Tensor -> Float32Array
    const TensorLatents = new Tensor('float32', jsLatents, [1, 4, 64, 64])               // Float32Array -> Tensor

    const jsLatentsMod = latentsMod.dataSync()                                                 
    const TensorLatentsMod = new Tensor('float32', jsLatentsMod, [1, 4, 64, 64])               

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

    const denoised = await processTimesteps(this.unet, this.scheduler, TensorLatents, promptEmbeds)
    const images = await this.makeImages(denoised)
    
    this.scheduler.setTimesteps(input.numInferenceSteps || 5)

    const denoisedMod = await processTimesteps(this.unet, this.scheduler, TensorLatentsMod, promptEmbeds)
    const imagesMod = await this.makeImages(denoisedMod)
    return [images, imagesMod];
  }
}
