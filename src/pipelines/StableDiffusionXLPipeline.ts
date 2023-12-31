import { PNDMScheduler, PNDMSchedulerConfig } from '@/schedulers/PNDMScheduler'
import { CLIPTokenizer } from '../tokenizers/CLIPTokenizer'
import { cat, randomNormalTensor } from '@/util/Tensor'
import { Tensor } from '@xenova/transformers'
import { dispatchProgress, loadModel, PretrainedOptions, ProgressCallback, ProgressStatus, sessionRun } from './common'
import { getModelFile, getModelJSON } from '../hub'
import { Session } from '../backends'
import { GetModelFileOptions } from '@/hub/common'
import { SchedulerConfig } from '@/schedulers/SchedulerBase'
import { PipelineBase } from '@/pipelines/PipelineBase'
import { LCMScheduler } from '@/schedulers/LCMScheduler'

export interface StableDiffusionXLInput {
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

export class StableDiffusionXLPipeline extends PipelineBase {
  public textEncoder2: Session
  public tokenizer2: CLIPTokenizer
  declare scheduler: PNDMScheduler

  constructor (
    unet: Session,
    vaeDecoder: Session,
    textEncoder: Session,
    textEncoder2: Session,
    tokenizer: CLIPTokenizer,
    tokenizer2: CLIPTokenizer,
    scheduler: PNDMScheduler,
  ) {
    super()
    this.unet = unet
    this.vaeDecoder = vaeDecoder
    this.textEncoder = textEncoder
    this.textEncoder2 = textEncoder2
    this.tokenizer = tokenizer
    this.tokenizer2 = tokenizer2
    this.scheduler = scheduler
  }

  static createScheduler (config: PNDMSchedulerConfig) {
    return new PNDMScheduler(
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

    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer' })
    const tokenizer2 = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer_2' })

    const unet = await loadModel(
      modelRepoOrPath,
      'unet/model.onnx',
      opts,
    )
    const textEncoder2 = await loadModel(modelRepoOrPath, 'text_encoder_2/model.onnx', opts)
    const textEncoder = await loadModel(modelRepoOrPath, 'text_encoder/model.onnx', opts)
    const vae = await loadModel(modelRepoOrPath, 'vae_decoder/model.onnx', opts)

    const schedulerConfig = await getModelJSON(modelRepoOrPath, 'scheduler/scheduler_config.json', true, opts)
    const scheduler = StableDiffusionXLPipeline.createScheduler(schedulerConfig)

    await dispatchProgress(opts.progressCallback, {
      status: ProgressStatus.Ready,
    })
    return new StableDiffusionXLPipeline(unet, vae, textEncoder, textEncoder2, tokenizer, tokenizer2, scheduler)
  }

  async encodePromptXl (prompt: string, tokenizer: CLIPTokenizer, textEncoder: Session) {
    const tokens = tokenizer(
      prompt,
      {
        return_tensor: false,
        padding: true,
        max_length: tokenizer.model_max_length,
        return_tensor_dtype: 'int32',
      },
    )

    const inputIds = tokens.input_ids
    const tensor = new Tensor('int32', Int32Array.from(inputIds.flat()), [1, inputIds.length])
    // @ts-ignore
    const result = await sessionRun(textEncoder, { input_ids: tensor })
    const {
      last_hidden_state: lastHiddenState,
      pooler_output: poolerOutput,
      // hidden_states: hiddenStates,
      'hidden_states.11': hiddenStates,
    } = result
    return { lastHiddenState, poolerOutput, hiddenStates }
  }

  async getPromptEmbedsXlClassifierFree (prompt: string, negativePrompt: string|undefined) {
    const promptEmbeds = await this.encodePromptXl(prompt, this.tokenizer, this.textEncoder)
    const negativePromptEmbeds = await this.encodePromptXl(negativePrompt || '', this.tokenizer, this.textEncoder)

    const promptEmbeds2 = await this.encodePromptXl(prompt, this.tokenizer2, this.textEncoder2)
    const negativePromptEmbeds2 = await this.encodePromptXl(negativePrompt || '', this.tokenizer2, this.textEncoder2)

    return {
      hiddenStates: cat([
        cat([negativePromptEmbeds.hiddenStates, negativePromptEmbeds2.hiddenStates], -1),
        cat([promptEmbeds.hiddenStates, promptEmbeds2.hiddenStates], -1),
      ]),
      textEmbeds: cat([randomNormalTensor(negativePromptEmbeds2.lastHiddenState.dims), randomNormalTensor(promptEmbeds2.lastHiddenState.dims)]),
    }
  }

  async getPromptEmbedsXl (prompt: string) {
    const promptEmbeds = await this.encodePromptXl(prompt, this.tokenizer, this.textEncoder)
    const promptEmbeds2 = await this.encodePromptXl(prompt, this.tokenizer2, this.textEncoder2)

    return {
      hiddenStates: cat([promptEmbeds.hiddenStates, promptEmbeds2.hiddenStates], -1),
      textEmbeds: promptEmbeds2.lastHiddenState,
    }
  }

  getTimeEmbeds (width: number, height: number, doClassifierFreeGuidance: boolean) {
    if (doClassifierFreeGuidance) {
      return new Tensor(
        'float32',
        Float32Array.from([height, width, 0, 0, height, width, height, width, 0, 0, height, width]),
        [2, 6],
      )
    }

    return new Tensor(
      'float32',
      Float32Array.from([height, width, 0, 0, height, width]),
      [1, 6],
    )
  }

  async run (input: StableDiffusionXLInput) {
    const width = input.width || 1024
    const height = input.height || 1024
    const batchSize = 1
    const guidanceScale = input.guidanceScale || 5
    const seed = input.seed || ''

    this.scheduler.setTimesteps(input.numInferenceSteps || 5)

    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.EncodingPrompt,
    })

    const doClassifierFreeGuidance = guidanceScale > 1
    const promptEmbeds = doClassifierFreeGuidance
      ? await this.getPromptEmbedsXlClassifierFree(input.prompt, input.negativePrompt)
      : await this.getPromptEmbedsXl(input.prompt)

    const latentShape = [batchSize, 4, width / 8, height / 8]
    let latents = randomNormalTensor(latentShape, undefined, undefined, 'float32', seed) // Normal latents used in Text-to-Image
    let denoised: Tensor
    const timesteps = this.scheduler.timesteps.data
    let humanStep = 1
    let cachedImages: Tensor[]|null = null

    const timeIds = this.getTimeEmbeds(width, height, doClassifierFreeGuidance)
    const hiddenStates = promptEmbeds.hiddenStates
    const textEmbeds = promptEmbeds.textEmbeds

    for (const step of timesteps) {
      const timestep = new Tensor(new Float32Array([step]))
      await dispatchProgress(input.progressCallback, {
        status: ProgressStatus.RunningUnet,
        unetTimestep: humanStep,
        unetTotalSteps: timesteps.length,
      })
      const latentInput = doClassifierFreeGuidance ? cat([latents, latents.clone()]) : latents

      const noise = await this.unet.run(
        {
          sample: latentInput,
          timestep,
          encoder_hidden_states: hiddenStates,
          text_embeds: textEmbeds,
          time_ids: timeIds,
        },
      )

      let noisePred = noise.out_sample
      if (doClassifierFreeGuidance) {
        const [noisePredUncond, noisePredText] = [
          noisePred.slice([0, 1]),
          noisePred.slice([1, 2]),
        ]
        noisePred = noisePredUncond.add(noisePredText.sub(noisePredUncond).mul(guidanceScale))
      }

      const schedulerOutput = this.scheduler.step(
        noisePred,
        step,
        latents,
      )

      latents = schedulerOutput
      denoised = schedulerOutput
      if (this.scheduler instanceof LCMScheduler) {
        latents = schedulerOutput[0]
        denoised = schedulerOutput[1]
      }

      if (input.runVaeOnEachStep) {
        await dispatchProgress(input.progressCallback, {
          status: ProgressStatus.RunningVae,
          unetTimestep: humanStep,
          unetTotalSteps: timesteps.length,
        })
        cachedImages = await this.makeImages(denoised)
      }
      humanStep++
    }

    if (input.runVaeOnEachStep) {
      return cachedImages!
    }

    return this.makeImages(denoised)
  }

  async release () {
    await super.release()
    return this.textEncoder2?.release()
  }
}
