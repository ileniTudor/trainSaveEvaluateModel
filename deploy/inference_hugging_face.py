from huggingface_hub import InferenceClient
client = InferenceClient()

image = client.text_to_image("An astronaut riding a horse on the moon.")
image.save("astronaut.png")  # 'image' is a PIL.Image object