import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
	def __init__(self, n_z, n_cat_code, n_conti_code, dim_cat_code):
		super(Generator, self).__init__()
		self.l1 = nn.Linear(n_z + n_cat_code*dim_cat_code + n_conti_code, 1024)
		self.bn1 = nn.BatchNorm1d(1024)

		self.l2 = nn.Linear(1024, 128*7*7)
		self.bn2 = nn.BatchNorm1d(128*7*7)
		# in_ch, out_ch, kernel, stride, padding		
		# conv (k=4) => 4x4 --> 1x1
		# convTranspose (k=4) => 1x1 --> 4x4
		# [-1, 128, 7, 7] -> [-1, 64, 14, 14]
		self.conv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1) 
		self.bn3 = nn.BatchNorm2d(64)
		# [-1, 64, 14, 14] -> [-1, 1, 28, 28]
		self.conv2 = nn.ConvTranspose2d(64, 1, 4, 2, 1)
		
	def forward(self, z):
		z = F.relu(self.bn1(self.l1(z)))
		z = F.relu(self.bn2(self.l2(z)))
		z = z.view(-1, 128, 7, 7)
		z = F.relu(self.bn3(self.conv1(z)))
#		z = F.tanh(self.conv2(z))
		z = self.conv2(z)
#		z = torch.squeeze(z, 1)
#		print(z.shape)
		return z

class Discriminator(nn.Module):
	def __init__(self, n_z, n_cat_code, n_conti_code, dim_cat_code):
		super(Discriminator, self).__init__()
		self.n_cat_code = n_cat_code
		self.n_conti_code = n_conti_code
		self.dim_cat_code = dim_cat_code
		self.lReLU = nn.LeakyReLU(0.1, inplace=True)
		# [-1, 1, 28, 28] -> [-1, 64, 14, 14]
		self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)
		# [-1, 64, 14, 14] -> [-1, 128, 7, 7]
		self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
		self.bn1 = nn.BatchNorm2d(128)
		# [-1, 128, 7, 7] -> [-1, 1024]
		#self.conv3 = nn.Conv2d(128, 1024, 7)
		#self.bn2 = nn.BatchNorm2d(1024)
		# [-1, 128*7*7] -> [-1, 1024]
		self.l1 = nn.Linear(128*7*7, 1024)
		self.bn2 = nn.BatchNorm1d(1024)
		# [-1, 1024] -> [-1, 1]
		self.D = nn.Linear(1024, 1)
		# [-1, 1024] -> [-1, 128]
		self.l_Q = nn.Linear(1024, 128)
		self.bn_Q = nn.BatchNorm1d(128)
		# [-1, 128] -> [-1, 1+10]
		self.Q = nn.Linear(128, self.n_cat_code*self.dim_cat_code+self.n_conti_code)

	def forward(self, z):
		z = self.lReLU(self.conv1(z))
		z = self.lReLU(self.bn1(self.conv2(z)))
		z = z.view(-1, 128*7*7)
		z = self.lReLU(self.bn2(self.l1(z)))
		z_d = self.D(z)
		z_q = self.Q(self.lReLU(self.bn_Q(self.l_Q(z))))
		# discriminator's output
		z_d = F.sigmoid(z_d.clone())
		# continuous latent codes = values
		# categorical code output
		z_q[:, self.n_cat_code-1 : self.n_cat_code*self.dim_cat_code] = F.softmax(z_q[:, self.n_cat_code : self.n_cat_code*self.dim_cat_code+1].clone())
		out = torch.cat((z_d, z_q), 1)
		return out
