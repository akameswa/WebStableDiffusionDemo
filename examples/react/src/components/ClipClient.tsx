import axios from 'axios';

export async function RetrieveImages(prompt: string, num_images: number) {
    const results = await axios.post(
        "https://knn.laion.ai/knn-service",
        {
            text:prompt,
            image: null,
            image_url: null,
            embedding_input: null,
            deduplicate: true,
            use_safety_model: true,
            use_violence_detector: true,
            indice_name: "laion5B-L-14",
            use_mclip: false,
            aesthetic_score: 9,
            aesthetic_weight: 0.5,
            modality: "image",
            num_images: num_images,
        }
    );

    const url_and_captions = [];

    for (let result of results.data) {
        const caption = result["caption"];
        const url = result["url"];

        try {
            const response = await axios.get(url, { responseType: 'arraybuffer' });
            if (response.status === 200) {
                url_and_captions.push({ url, caption });
            }
        }catch(e){
            console.log(e);
        }
    }

    return url_and_captions;
}