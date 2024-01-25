import { dispatchProgress, loadModel, PretrainedOptions, ProgressCallback, ProgressStatus } from '@/pipelines/common'
import { Session } from '@/backends'
import { LCMScheduler, LCMSchedulerConfig } from '@/schedulers/LCMScheduler'
import { GetModelFileOptions } from '@/hub/common'
import { CLIPTokenizer } from '@/tokenizers/CLIPTokenizer'
import { getModelJSON } from '@/hub'
import { Tensor } from '@xenova/transformers'
import { PipelineBase } from '@/pipelines/PipelineBase'
import * as tf from '@tensorflow/tfjs';

export interface SeedInput {
  prompt: string
  numImages: number
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

export class SeedPipeline extends PipelineBase {
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
    const scheduler = SeedPipeline.createScheduler(schedulerConfig)

    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer' })
    await dispatchProgress(opts.progressCallback, {
      status: ProgressStatus.Ready,
    })
    return new SeedPipeline(unet, vae, vaeEncoder, textEncoder, tokenizer, scheduler)
  }

  async release () {
    await this.unet?.release()
    await this.vaeDecoder?.release()
    await this.vaeEncoder?.release()
    await this.textEncoder?.release()
  }

  async run (input: SeedInput) {
    this.scheduler.setTimesteps(input.numInferenceSteps || 5)

    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.EncodingPrompt,
    })

    const promptEmbeds = await this.encodePrompt(input.prompt)
    
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
      let latents = tf.randomNormal([4, 64, 64], 0, 1, 'float32', i)      // mean: 0, std: 1 as per codebase randomNormalTensor
      let batchJS = latents.dataSync()                                    // tf.Tensor -> Float32Array
      let batchTensor = new Tensor('float32', batchJS, [1, 4, 64, 64])    // Float32Array -> Tensor
      let denoised = await processTimesteps(this.unet, this.scheduler, batchTensor, promptEmbeds)
      images.push(await this.makeImages(denoised))
    }

    return [images];
  }
}
