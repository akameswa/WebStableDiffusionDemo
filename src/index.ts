// import 'module-alias/register.js'
import browserCache from '@/hub/browser'
import { setCacheImpl } from '@/hub'

export * from './pipelines/StableDiffusionPipeline'
export * from './pipelines/StableDiffusionXLPipeline'
export * from './pipelines/LatentConsistencyModelPipeline'
export * from './pipelines/SimilarImagePipeline'
export * from './pipelines/PromptInterpolationPipeline'
export * from './pipelines/DiffusionPipeline'
export * from './pipelines/common'
export * from './hub'
export * from './backends'
export { setModelCacheDir } from '@/hub/browser'

setCacheImpl(browserCache)
