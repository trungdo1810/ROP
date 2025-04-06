import gdown

url = "https://drive.google.com/file/d/1AF_vaqjV_advfrVjN3VCukGLpGd5D5-o/view?usp=sharing"
output = "dataset.zip"

gdown.download(url, output, quiet=False)

print("Finished!")