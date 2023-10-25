import { PretrainedOptions } from '@/pipelines/common'
import { GetModelFileOptions } from '@/hub/common'
import { getModelJSON } from '@/hub'
import { StableDiffusionPipeline } from '@/pipelines/StableDiffusionPipeline'
import { StableDiffusionXLPipeline } from '@/pipelines/StableDiffusionXLPipeline'

export class DiffusionPipeline {
  static async fromPretrained (modelRepoOrPath: string, options?: PretrainedOptions) {
    const opts: GetModelFileOptions = {
      ...options,
    }

    const index = await getModelJSON(modelRepoOrPath, 'model_index.json', true, opts)

    switch (index['_class_name']) {
      case 'StableDiffusionPipeline':
      case 'OnnxStableDiffusionPipeline':
        return StableDiffusionPipeline.fromPretrained(modelRepoOrPath, options)
      case 'StableDiffusionXLPipeline':
        return StableDiffusionXLPipeline.fromPretrained(modelRepoOrPath, options)
      default:
        throw new Error(`Unknown pipeline type ${index['_class_name']}`)
    }
  }
}