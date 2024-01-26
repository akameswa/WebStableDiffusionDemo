import { dispatchProgress, loadModel, PretrainedOptions, ProgressCallback, ProgressStatus } from '@/pipelines/common'
import { Session } from '@/backends'
import { LCMScheduler, LCMSchedulerConfig } from '@/schedulers/LCMScheduler'
import { GetModelFileOptions } from '@/hub/common'
import { CLIPTokenizer } from '@/tokenizers/CLIPTokenizer'
import { getModelJSON } from '@/hub'
import { Tensor } from '@xenova/transformers'
import { PipelineBase } from '@/pipelines/PipelineBase'
import * as tf from '@tensorflow/tfjs';

export interface AnimationInput {
  prompt: string
  numImages?: number
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

export class AnimationPipeline extends PipelineBase {
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
    const scheduler = AnimationPipeline.createScheduler(schedulerConfig)

    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer' })
    await dispatchProgress(opts.progressCallback, {
      status: ProgressStatus.Ready,
    })
    return new AnimationPipeline(unet, vae, vaeEncoder, textEncoder, tokenizer, scheduler)
  }

  async release () {
    await this.unet?.release()
    await this.vaeDecoder?.release()
    await this.vaeEncoder?.release()
    await this.textEncoder?.release()
  }

  async run (input: AnimationInput) {
    const seed = input.seed || '69'
    this.scheduler.setTimesteps(input.numInferenceSteps || 5)

    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.EncodingPrompt,
    })

    const promptEmbeds = await this.encodePrompt(input.prompt)
    
    async function processTimesteps (unet: Session, scheduler: LCMScheduler, vaeDecoder: Session, latents: Tensor, promptEmbeds: Tensor) {
      let timesteps = scheduler.timesteps.data
      let humanStep = 1
      let denoised: Tensor
      const denoisedImages = []

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
        
        let denoisedX = denoised.add(latents.mul(timesteps.length - humanStep - 1)).div(0.18215)

        let decoded = await vaeDecoder.run(
          { latent_sample: denoisedX },
        )
        let decodedImages = decoded.sample
          .div(2)
          .add(0.5)
          .clipByValue(0, 1)         

        denoisedImages.push([decodedImages])
      }
      humanStep++

      await dispatchProgress(input.progressCallback, {
        status: ProgressStatus.Done,
      })
      return [denoisedImages];
    }
    const latents = tf.randomNormal([4, 64, 64], 0, 1, 'float32', parseInt(seed))      // mean: 0, std: 1 as per codebase randomNormalTensor
    const batchJS = latents.dataSync()                                                 // tf.Tensor -> Float32Array
    const batchTensor = new Tensor('float32', batchJS, [1, 4, 64, 64])                 // Float32Array -> Tensor
    const images = await processTimesteps(this.unet, this.scheduler, this.vaeDecoder, batchTensor, promptEmbeds)

    return images;
  }
}
