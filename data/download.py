import os
import requests
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

data_dir = "."

if __name__=="__main__":
	###############
	#### MNIST ####
	###############
	transform = transforms.Compose([transforms.Resize(28), transforms.CenterCrop(28), transforms.ToTensor()])
	if not (os.path.exists(os.path.join(data_dir, "MNIST"))):
		mnist = datasets.MNIST(os.path.join(data_dir), train="train", download=True, transform=transform)
	else:
		print("Dataset exists at %s" % (os.path.join(data_dir, "MNIST")))

	##############
	#### SVHN ####
	##############
	transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
	if not (os.path.exists(os.path.join(data_dir, "SVHN"))):
		_ = os.mkdir(os.path.join(data_dir, "SVHN"))
		for name in ["train", "test", "extra"]:
			r = requests.get("http://ufldl.stanford.edu/housenumbers/%s_32x32.mat" % name)
			f = open(os.path.join(data_dir, "SVHN", "%s_32x32.mat" % name), "wb")
			for chunk in r.iter_content(chunk_size=512*1024):
				if chunk: f.write(chunk)
			f.close()
		#svhn = datasets.SVHN(os.path.join(data_dir, "SVHN"), split="train", download=True, transform=transform)
	else:
		print("Dataset exists at %s" % (os.path.join(data_dir, "SVHN")))

	################
	#### CelebA ####
	################
	transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	if not (os.path.exists(os.path.join(data_dir, "CelebA"))):
		#r = requests.get("https://www.dropbox.com/s/e0ig4nf1v94hyj8/CelebA_128crop_FD.zip?dl=0")
		#f = open(os.path.join(data_dir, "CelebA", "CelebA_128crop_FD.zip"), "wb")
		#for chunk in r.iter_content(chunk_size=512*1024):
		#	if chunk: f.write(chunk)
		#f.close()
		celeba = datasets.CelebA(os.path.join(data_dir), split="train", download=True, transform=transform)
	else:
		print("Dataset exists at %s" % (os.path.join(data_dir, "CelebA")))


