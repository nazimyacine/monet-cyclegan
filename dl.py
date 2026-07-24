import urllib.request, zipfile, os

url = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip"
print("Téléchargement...")
urllib.request.urlretrieve(url, "monet2photo.zip")

print("Extraction...")
with zipfile.ZipFile("monet2photo.zip", "r") as z:
    z.extractall("dataset")

os.remove("monet2photo.zip")
print("Terminé ! Dossiers créés dans ./dataset/")