import React, { useEffect, useRef, useState } from 'react'
import './App.css';
import {
  DiffusionPipeline,
  ProgressCallback,
  ProgressCallbackPayload,
  setModelCacheDir,
  StableDiffusionPipeline,
  StableDiffusionXLPipeline,
  LatentConsistencyModelPipeline,
  CirclePipeline,
  PromptInterpolationPipeline
} from '@aislamov/diffusers.js'
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import Stack from '@mui/material/Stack';
import Grid from '@mui/material/Grid';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import { Checkbox, FormControl, InputLabel, MenuItem, Select, Tab, Tabs, ImageList, ImageListItem, ImageListItemBar } from '@mui/material';
import { FormControlLabel } from '@mui/material';
import { BrowserFeatures, hasFp16 } from './components/BrowserFeatures'
import { Tensor } from '@xenova/transformers'
import cv from '@techstark/opencv-js'
import { StableDiffusionControlNetPipeline } from '../../../dist/pipelines/StableDiffusionControlNetPipeline';
import { CustomTabPanel } from './components/TabPanel';
import { RetrieveImages } from './components/ClipClient';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});

interface SelectedPipeline {
  name: string
  repo: string
  revision: string
  fp16: boolean
  steps: number
  hasImg2Img: boolean
  hasControlNet: boolean
  width: number
  height: number
}

const pipelines = [
  {
    name: 'LCM TinySD FP16 (1.1GB)',
    repo: 'akameswa/lcm-tiny-sd-onnx-fp16',
    revision: 'main',
    fp16: true,
    width: 512,
    height: 512,
    steps: 4,
    hasImg2Img: false,
    hasControlNet: false
  },
]

function App() {
  const [hasF16, setHasF16] = useState<boolean>(false);
  const [selectedPipeline, setSelectedPipeline] = useState<SelectedPipeline|undefined>(pipelines[0]);
  const [modelState, setModelState] = useState<'none'|'loading'|'ready'|'inferencing'>('none');
  const [prompt, setPrompt] = useState('Self-portrait oil painting, a beautiful cyborg with golden hair, 8k');
  const [promptA, setPromptA] = useState('Self-portrait oil painting, a beautiful man with golden hair, 8k');
  const [promptB, setPromptB] = useState('Self-portrait oil painting, a beautiful woman with golden hair, 8k');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [inferenceSteps, setInferenceSteps] = useState(20);
  const [guidanceScale, setGuidanceScale] = useState(1);
  const [seed, setSeed] = useState('69');
  const [status, setStatus] = useState('Ready');
  const pipeline = useRef<StableDiffusionXLPipeline|StableDiffusionPipeline|StableDiffusionControlNetPipeline|LatentConsistencyModelPipeline|SeedPipeline|CirclePipeline|PromptInterpolationPipeline|null>(null);
  const [img2img, setImg2Img] = useState(false);
  const [inputImage, setInputImage] = useState<Float32Array>();
  const [strength, setStrength] = useState(0.8);
  const [controlNetImage, setControlNetImage] = useState<Float32Array>();
  const [runVaeOnEachStep, setRunVaeOnEachStep] = useState(false);
  const [tab, setTab] = useState(0);
  const [urlCaptions, setUrlCaptions] = useState<any []>([]);
  const [numImages, setNumImages] = useState(5);
  const [differentiation, setDifferentiation] = useState(1);
  const [urls, seturls] = useState<any []>([]);

  useEffect(() => {
    setModelCacheDir('models')
    hasFp16().then(v => {
      setHasF16(v)
      if (v === false) {
        setSelectedPipeline(pipelines.find(p => p.fp16 === false))
      }
    })
  }, [])

  useEffect(() => {
    setInferenceSteps(selectedPipeline?.steps || 20)
  }, [selectedPipeline])

  interface link {
    canvasUrl: string
    index: string
  }
  const links: link[] = []                       // contains poked and unpoked images

  const drawImage = async (image: Tensor, index: string) => {
    const canvas = document.getElementById('canvas') as HTMLCanvasElement
    if (canvas) {
      const data = await image.toImageData({ tensorLayout: 'NCHW', format: 'RGB' });
      canvas.getContext('2d')!.putImageData(data, 0, 0);
      if (index === 'Poked') {
        const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
        ctx.beginPath();
        ctx.lineWidth = 5;
        ctx.strokeStyle = "white";
        ctx.rect(0, 0, 128, 128);
        ctx.stroke();
      } 
      const canvasUrl = await canvas.toDataURL();
      links.push({canvasUrl, index});
    }
  }

  const progressCallback = async (info: ProgressCallbackPayload) => {
    if (info.statusText) {
      setStatus(info.statusText)
    }

    if (info.images) {
      // @ts-ignore
      await drawImage(info.images[0])
    }
  }

  const loadModel = async () => {
    if (!selectedPipeline) {
      return
    }
    setModelState('loading')
    try {
      if(tab==0){
        if (pipeline.current) {
          // @ts-ignore
          pipeline.current.release()
        }
        pipeline.current = await LatentConsistencyModelPipeline.fromPretrained(
          selectedPipeline.repo,
          {
            revision: selectedPipeline?.revision,
            progressCallback
          }
        )
      }else if(tab==1){
        if (pipeline.current) {
          // @ts-ignore
          pipeline.current.release()
        }
        pipeline.current = await PromptInterpolationPipeline.fromPretrained(
          selectedPipeline.repo,
          {
            revision: selectedPipeline?.revision,
            progressCallback
          }
        )
      }else if(tab==2){
        if (pipeline.current) {
          // @ts-ignore
          pipeline.current.release()
        }
        pipeline.current = await CirclePipeline.fromPretrained(
          selectedPipeline.repo,
          {
            revision: selectedPipeline?.revision,
            progressCallback
          }
        )
      }else if(tab==3){
          selectedPipeline.repo,
          {
            revision: selectedPipeline?.revision,
            progressCallback
          }
        )
      }else{
        if (pipeline.current) {
          // @ts-ignore
          pipeline.current.release()
        }
        pipeline.current = await DiffusionPipeline.fromPretrained(
          selectedPipeline.repo,
          {
            revision: selectedPipeline?.revision,
            progressCallback
          }
        )
      }
      setModelState('ready')
    } catch (e) {
      alert(e)
      console.error(e)
    }
  }

  /**
   * Extracts the RGB data from an RGBA image array.
   * 
   * @param d RGBA image array.
   * @param normalize Normalizes the image array to [-1,1] if true. Set to true for img2img and false for controlnet.
   * @returns RGB Float32Array.
   */
  function getRgbData(d: Uint8ClampedArray, normalize=true) {
    let rgbData: any = [[], [], []]; // [r, g, b]
    // remove alpha and put into correct shape:
    for(let i = 0; i < d.length; i += 4) {
        let x = (i/4) % 512;
        let y = Math.floor((i/4) / 512)
        if(!rgbData[0][y]) rgbData[0][y] = [];
        if(!rgbData[1][y]) rgbData[1][y] = [];
        if(!rgbData[2][y]) rgbData[2][y] = [];
        rgbData[0][y][x] = normalize ? (d[i+0]/255) * 2 - 1 : (d[i+0]/255);
        rgbData[1][y][x] = normalize ? (d[i+1]/255) * 2 - 1 : (d[i+1]/255);
        rgbData[2][y][x] = normalize ? (d[i+2]/255) * 2 - 1 : (d[i+2]/255);
    }
    rgbData = Float32Array.from(rgbData.flat().flat());
    return rgbData;
  }

  /**
   * Takes an input image and saves it to the corresponding state variable. 
   * The input image can be used either for the controlnet or img2img pipelines
   * which is determined by type.
   * 
   * @param e HTML file upload element.
   * @param type Pipeline of the input image.
   * @returns void
   */
  function uploadImage(e: any, type: 'controlnet'|'img2img') {
    if(!e.target.files[0]) {
      // No image uploaded
      return;
    }

    const uploadedImage = new Image(512, 512); // resize image to 512, 512
    const reader = new FileReader();
    // On file read loadend
    reader.addEventListener('loadend', function(file: any) {
      // On image load
      uploadedImage.addEventListener('load', function() {
        if (type == 'img2img') {
          const imageCanvas = document.createElement('canvas');
          imageCanvas.width = uploadedImage.width;
          imageCanvas.height = uploadedImage.height;
          const imgCtx = imageCanvas.getContext('2d') as CanvasRenderingContext2D;
          imgCtx.drawImage(uploadedImage, 0, 0, uploadedImage.width, uploadedImage.height);
          const imageData = imgCtx.getImageData(0, 0, uploadedImage.width, uploadedImage.height).data;

          const rgb_array = getRgbData(imageData);
          setInputImage(rgb_array);
        }
        else if(type == 'controlnet') {
          // For now only Canny Edge Detection is available
          const cvImg = cv.imread(uploadedImage); // RGBA Image | 4 Channels
          const imgGray = new cv.Mat();
          cv.cvtColor(cvImg, imgGray, cv.COLOR_RGBA2GRAY); // Gray Image | 1 Channel
          const imgCanny = new cv.Mat();
          cv.Canny(imgGray, imgCanny, 100, 200, 3, false); // Canny Image | 1 Channel
          const rgbaCanny = new cv.Mat();
          cv.cvtColor(imgCanny, rgbaCanny, cv.COLOR_GRAY2RGBA, 0); // RGBA Canny Image | 4 Channels

          /**
           * The canny data can be accessed as so:
           * cannyEdges.data -> UInt8Array
           * cannyEdges.data8S -> Int8Array
           * cannyEdges.data16S -> Int16Array
           * cannyEdges.data16U -> UInt16Array
           * cannyEdges.data32F -> Float32Array
           * cannyEdges.data32S -> Int32Array
           * cannyEdges.data64F -> Float64Array
           */

          const rgbCanny = getRgbData(Uint8ClampedArray.from(rgbaCanny.data), false);
          setControlNetImage(rgbCanny);
          cvImg.delete();imgGray.delete();imgCanny.delete();rgbaCanny.delete();
        }
      });
      uploadedImage.src = file.target.result;
    });
    reader.readAsDataURL(e.target.files[0]);
  }

  const runInference = async () => {
    if (!pipeline.current) {
      return
    }
    setModelState('inferencing')

    if(pipeline.current instanceof LatentConsistencyModelPipeline){
        const [images, imagesMod] = await pipeline.current.run({
          prompt: prompt,
          negativePrompt: negativePrompt,
          numInferenceSteps: inferenceSteps,
          guidanceScale: guidanceScale,
          seed: seed,
          width: selectedPipeline?.width,
          height: selectedPipeline?.height,
          runVaeOnEachStep,
          progressCallback,
          img2imgFlag: img2img,
          inputImage: inputImage,
          strength: strength,
        })
        await drawImage(images[0], 'Normal')
        await drawImage(imagesMod[0], 'Poked')
              seturls(links)
        setModelState('ready')
    }else if(pipeline.current instanceof PromptInterpolationPipeline){
        const [images] = await pipeline.current.run({
          promptA: promptA,
          promptB: promptB,
          numImages: numImages,
          numInferenceSteps: inferenceSteps,
          seed: seed,
          width: selectedPipeline?.width,
          height: selectedPipeline?.height,
          progressCallback,
        })
        for(let i = 0; i < numImages; i++) {
          await drawImage(images[i][0], (i+1).toString())
        }
        seturls(links)
        setModelState('ready')
    }
    else if(pipeline.current instanceof CirclePipeline){
        const [images] = await pipeline.current.run({
          prompt: prompt,
          numImages: numImages,
          differentiation: differentiation,
          numInferenceSteps: inferenceSteps,
          seed: seed,
          width: selectedPipeline?.width,
          height: selectedPipeline?.height,
          progressCallback,
        })
        for(let i = 0; i < numImages; i++) {
          await drawImage(images[i][0], (i+1).toString())
        }
        seturls(links)
        setModelState('ready')
    }
        progressCallback,
      })
      for(let i = 0; i < numImages; i++) {
        await drawImage(images[i][0], (i+1).toString())
      }
      seturls(links)
      setModelState('ready')
  }
  }

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline enableColorScheme={true} />
      <Container>
        <BrowserFeatures />
        <Tabs value={tab} onChange={(e,v) => {
          setTab(v)
          setModelState('none')
        }} centered>
            <Tab label="Poke" />
            <Tab label="Interpolate" />
            <Tab label="Circle" />
            <Tab label="Dataset Peek" />
          </Tabs>
        <CustomTabPanel value={tab} index={0}>
        <Box sx={{ bgcolor: '#282c34' }} pt={4} pl={3} pr={3} pb={4}>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Stack spacing={2}>
                <TextField
                  label="Prompt"
                  variant="standard"
                  disabled={modelState != 'ready'}
                  onChange={(e) => setPrompt(e.target.value)}
                  value={prompt}
                />
                {/* <TextField
                  label="Negative Prompt"
                  variant="standard"
                  disabled={modelState != 'ready'}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  value={negativePrompt}
                /> */}
                <TextField
                  label="Number of inference steps"
                  variant="standard"
                  type='number'
                  disabled={modelState != 'ready'}
                  onChange={(e) => setInferenceSteps(parseInt(e.target.value))}
                  value={inferenceSteps}
                />
                {/* <TextField
                  label="Guidance Scale. Controls how similar the generated image will be to the prompt."
                  variant="standard"
                  type='number'
                  InputProps={{ inputProps: { min: 1, max: 20, step: 0.5 } }}
                  // disabled={modelState != 'ready'}
                  onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
                  value={guidanceScale}
                /> */}
                <TextField
                  label="Seed"
                  variant="standard"
                  disabled={modelState != 'ready'}
                  onChange={(e) => setSeed(e.target.value)}
                  value={seed}
                />
                {selectedPipeline?.hasControlNet &&
                  (
                    <>
                      <label htmlFor="upload_controlnet_image">Upload Image for ControlNet Pipeline:</label>
                      <TextField
                        id="upload_controlnet_image"
                        inputProps={{accept:"image/*"}}
                        type={"file"}
                        disabled={modelState != 'ready'}
                        onChange={(e) => uploadImage(e, "controlnet")}
                      />
                  </>
                )}
                {selectedPipeline?.hasImg2Img &&
                  (
                    <>
                      <FormControlLabel
                        label="Check if you want to use the Img2Img pipeline"
                        control={<Checkbox
                          disabled={modelState != 'ready'}
                          onChange={(e) => setImg2Img(e.target.checked)}
                          checked={img2img}
                        />}
                      />
                      <label htmlFor="upload_image">Upload Image for Img2Img Pipeline:</label>
                      <TextField
                        id="upload_image"
                        inputProps={{accept:"image/*"}}
                        type={"file"}
                        disabled={!img2img}
                        onChange={(e) => uploadImage(e, 'img2img')}
                      />
                      <TextField
                        label="Strength (Noise to add to input image). Value ranges from 0 to 1"
                        variant="standard"
                        type='number'
                        InputProps={{ inputProps: { min: 0, max: 1, step: 0.1 } }}
                        disabled={!img2img}
                        onChange={(e) => setStrength(parseFloat(e.target.value))}
                        value={strength}
                      />
                    </>
                )}
                {/* <FormControlLabel
                  label="Check if you want to run VAE after each step"
                  control={<Checkbox
                    disabled={modelState != 'ready'}
                    onChange={(e) => setRunVaeOnEachStep(e.target.checked)}
                    checked={runVaeOnEachStep}
                  />}
                /> */}
                <FormControl fullWidth>
                  <InputLabel id="demo-simple-select-label">Pipeline</InputLabel>
                    <Select
                      value={selectedPipeline?.name}
                      onChange={e => {
                        setSelectedPipeline(pipelines.find(p => e.target.value === p.name))
                        setModelState('none')
                      }}>
                      {pipelines.map(p => <MenuItem value={p.name} disabled={!hasF16 && p.fp16}>{p.name}</MenuItem>)}
                    </Select>
                </FormControl>
                {/* <p>Press the button below to download model. It will be stored in your browser cache.</p>
                <p>All settings above will become editable once model is downloaded.</p> */}
                <Button variant="outlined" onClick={loadModel} disabled={modelState != 'none'}>Load model</Button>
                <Button variant="outlined" onClick={runInference} disabled={modelState != 'ready'}>Run</Button>
                <p>{status}</p>
                {/* <p><a href={'https://github.com/dakenf'}>Follow me on GitHub</a></p> */}
              </Stack>
            </Grid>
            <Grid item xs={6}>
              <ImageList variant='quilted' style={{ border: '1px dashed #ccc'}} sx={{ width: 512, height: 512}} cols={1} gap={8}>
                  {urls.map((item) => {
                    return (
                      <ImageListItem key={item.index}>
                        <img src={item.canvasUrl} alt={item.index}/>
                        <ImageListItemBar
                          subtitle={item.index}
                        />
                      </ImageListItem>
                  )})}
                </ImageList>
              <canvas id={'canvas'} width={selectedPipeline?.width} height={selectedPipeline?.height} style={{ border: '1px dashed #ccc'}} hidden/>
            </Grid>
          </Grid>
        </Box>
        <Divider/>
        </CustomTabPanel>
        <CustomTabPanel value={tab} index={1}>
        <Box sx={{ bgcolor: '#282c34' }} pt={4} pl={3} pr={3} pb={4}>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Stack spacing={2}>
                <TextField
                  label="Prompt from"
                  variant="standard"
                  sx={{ width: '100%' }}
                  onChange={(e) => setPromptA(e.target.value)}
                  disabled={modelState != 'ready'}
                  value={promptA}
                />
                <TextField
                  label="Prompt to"
                  variant="standard"
                  sx={{ width: '100%' }}
                  onChange={(e) => setPromptB(e.target.value)}
                  disabled={modelState != 'ready'}
                  value={promptB}
                />
                <TextField
                  label="Number of images"
                  variant="standard"
                  type='number'
                  sx={{ width: '100%' }}
                  InputProps={{ inputProps: { min: 1, max: 10, step: 1 } }}
                  onChange={(e) => setNumImages(parseInt(e.target.value))}
                  disabled={modelState != 'ready'}
                  value={numImages}
                />
                <TextField
                  label="Number of inference steps"
                  variant="standard"
                  type='number'
                  sx={{ width: '100%' }}
                  onChange={(e) => setInferenceSteps(parseInt(e.target.value))}
                  disabled={modelState != 'ready'}
                  value={inferenceSteps}
                />
                <TextField
                  label="Seed"
                  variant="standard"
                  sx={{ width: '100%' }}
                  onChange={(e) => setSeed(e.target.value)}
                  disabled={modelState != 'ready'}
                  value={seed}
                />
                <FormControl fullWidth>
                  <InputLabel id="demo-simple-select-label">Pipeline</InputLabel>
                    <Select
                      value={selectedPipeline?.name}
                      onChange={e => {
                        setSelectedPipeline(pipelines.find(p => e.target.value === p.name))
                        setModelState('none')
                      }}>
                      {pipelines.map(p => <MenuItem value={p.name} disabled={!hasF16 && p.fp16}>{p.name}</MenuItem>)}
                    </Select>
                </FormControl>
                <p> <Button variant="outlined" onClick={loadModel} disabled={modelState != 'none'} fullWidth>Load model</Button> </p>
                <p> <Button variant="outlined" onClick={runInference} disabled={modelState != 'ready'} fullWidth>Run</Button> </p>
                <p>{status}</p>
              </Stack> 
            </Grid>
            <Grid item xs={6}>
                <ImageList variant='quilted' style={{ border: '1px dashed #ccc'}} sx={{ width: 512, height: 512}} cols={1} gap={8}>
                  {urls.map((item) => {
                    return (
                      <ImageListItem key={item.index}>
                        <img src={item.canvasUrl} alt={item.index}/>
                        <ImageListItemBar
                          subtitle={item.index}
                        />
                      </ImageListItem>
                  )})}
                </ImageList>
              <canvas id={'canvas'} width={selectedPipeline?.width} height={selectedPipeline?.height} style={{ border: '1px dashed #ccc'}} hidden/>
            </Grid>
          </Grid>
          </Box>
        </CustomTabPanel>
        <CustomTabPanel value={tab} index={2}>
          <Box sx={{ bgcolor: '#282c34' }} pt={4} pl={3} pr={3} pb={4}>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Stack spacing={2}>
                <TextField
                  label="Prompt"
                  variant="standard"
                  sx={{ width: '100%' }}
                  onChange={(e) => setPrompt(e.target.value)}
                  disabled={modelState != 'ready'}
                  value={prompt}
                />
                <TextField
                  label="Number of images"
                  variant="standard"
                  type='number'
                  sx={{ width: '100%' }}
                  InputProps={{ inputProps: { min: 1, max: 10, step: 1 } }}
                  onChange={(e) => setNumImages(parseInt(e.target.value))}
                  disabled={modelState != 'ready'}
                  value={numImages}
                />
                <TextField
                  label="Differentiation"
                  variant="standard"
                  type='number'
                  sx={{ width: '100%' }}
                  InputProps={{ inputProps: { min: 0, max: 1} }}
                  onChange={(e) => setDifferentiation(parseFloat(e.target.value))}
                  disabled={modelState != 'ready'}
                  value={differentiation}
                />
                <TextField
                  label="Number of inference steps"
                  variant="standard"
                  type='number'
                  sx={{ width: '100%' }}
                  disabled={modelState != 'ready'}
                  onChange={(e) => setInferenceSteps(parseInt(e.target.value))}
                  value={inferenceSteps}
                />
                <TextField
                  label="Seed"
                  variant="standard"
                  sx={{ width: '100%' }}
                  disabled={modelState != 'ready'}
                  onChange={(e) => setSeed(e.target.value)}
                  value={seed}
                />
                <FormControl fullWidth>
                  <InputLabel id="demo-simple-select-label">Pipeline</InputLabel>
                    <Select
                      value={selectedPipeline?.name}
                      onChange={e => {
                        setSelectedPipeline(pipelines.find(p => e.target.value === p.name))
                        setModelState('none')
                      }}>
                      {pipelines.map(p => <MenuItem value={p.name} disabled={!hasF16 && p.fp16}>{p.name}</MenuItem>)}
                    </Select>
                </FormControl>
                <p> <Button variant="outlined" onClick={loadModel} disabled={modelState != 'none'} fullWidth>Load model</Button> </p>
                <p> <Button variant="outlined" onClick={runInference} disabled={modelState != 'ready'} fullWidth>Run</Button> </p>
                <p>{status}</p>
              </Stack> 
            </Grid>
            <Grid item xs={6}>
                <ImageList variant='quilted' style={{ border: '1px dashed #ccc'}} sx={{ width: 512, height: 512}} cols={1} gap={8}>
                  {urls.map((item) => {
                    return (
                      <ImageListItem key={item.index}>
                        <img src={item.canvasUrl} alt={item.index}/>
                        <ImageListItemBar
                          subtitle={item.index}
                        />
                      </ImageListItem>
                  )})}
                </ImageList>
              <canvas id={'canvas'} width={selectedPipeline?.width} height={selectedPipeline?.height} style={{ border: '1px dashed #ccc'}} hidden/>
            </Grid>
          </Grid>
          </Box>
        </CustomTabPanel>
        <CustomTabPanel value={tab} index={3}>
            <Box sx={{ bgcolor: '#282c34' }} pt={4} pl={3} pr={3} pb={4}>
              <Stack spacing={2}>
              <TextField
                label="Prompt"
                variant="standard"
                sx={{ width: '100%' }}
                onChange={(e) => setPrompt(e.target.value)}
                value={prompt}
              />
              <TextField
                label="Number of images to retrieve"
                variant="standard"
                type='number'
                sx={{ width: '100%' }}
                InputProps={{ inputProps: { min: 0, max: 50, step: 1 } }}
                onChange={(e) => setNumImages(parseInt(e.target.value))}
                value={numImages}
              />
              <p>
                <Button variant="outlined" onClick={()=>{
                    RetrieveImages(prompt, numImages).then((item) => {
                      setUrlCaptions(item);
                    });
                  }}>Retrieve</Button>
              </p>
              <ImageList variant='quilted' style={{ border: '1px dashed #ccc'}} sx={{ width: '100%', height: 512}} cols={3} gap={8}>
                {urlCaptions.map((item, index) => (
                  <ImageListItem key={index}>
                    <img
                      crossOrigin='anonymous'
                      src={`${item.url}`}
                      loading="lazy"
                      alt={item.caption}
                    />
                    <ImageListItemBar
                      subtitle={item.caption}
                    />
                  </ImageListItem>
                ))}
              </ImageList>
              </Stack>
            </Box>
          </CustomTabPanel>
        <Stack alignItems={'center'}>
          <p>Built with <a href={"https://github.com/dakenf/diffusers.js"} target={"_blank"}>diffusers.js</a></p>
        </Stack>
      </Container>
    </ThemeProvider>
  );
}

export default App;
