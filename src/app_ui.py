import gradio as gr
import cv2

from image_utils import (
    load_image_from_url,
    prewitt_edge_detection,
    sobel,
    difference_of_gaussians
)

def generate(image_source, url=None, upload=None):
    if image_source == "URL":
        if not url:
            raise gr.Error("Please provide a URL when selecting the URL option")
        img = load_image_from_url(url)
    else:
        if upload is None:
            raise gr.Error("Please upload an image when selecting the upload option")
        img = upload

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(img, 100, 200)
    prewitt = prewitt_edge_detection(img)
    sobel_img = sobel(img)
    dog_img = difference_of_gaussians(img)

    return canny, prewitt, sobel_img, dog_img

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Edge Detection Comparison")
        gr.Markdown("Choose whether to provide an image URL or upload an image")

        with gr.Row():
            image_source = gr.Radio(
                choices=["Upload", "URL"],
                label="Image Source",
                value="Upload"
            )

        with gr.Row():
            with gr.Column(visible=True) as upload_col:
                upload = gr.Image(label="Upload Image")
            with gr.Column(visible=False) as url_col:
                url = gr.Textbox(label="Image URL")

        btn = gr.Button("Generate Edge Detection")

        with gr.Row():
            canny_output = gr.Image(label="Canny Edges")
            prewitt_output = gr.Image(label="Prewitt Edges")
            sobel_output = gr.Image(label="Sobel Edges")
            dog_output = gr.Image(label="DoG Edges")

        def toggle_inputs(image_source):
            if image_source == "Upload":
                return {upload_col: gr.Column(visible=True), url_col: gr.Column(visible=False)}
            else:
                return {upload_col: gr.Column(visible=False), url_col: gr.Column(visible=True)}

        image_source.change(
            fn=toggle_inputs,
            inputs=image_source,
            outputs=[upload_col, url_col]
        )

        btn.click(
            fn=generate,
            inputs=[image_source, url, upload],
            outputs=[canny_output, prewitt_output, sobel_output, dog_output]
        )

    return demo
