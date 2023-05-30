import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import src
import src.Model as m

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device",device)
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 50

disc = m.Discriminator(image_dim).to(device)
gen = m.Generator(z_dim,image_dim).to(device)

fixed_noise = torch.randn((batch_size,z_dim)).to(device)

transforms  = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5),(0.5))]
)

dataset = datasets.MNIST(root = "dataset/", transform = transforms,download = True)
loader = DataLoader(dataset,batch_size = batch_size, shuffle = True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_dx, (real_data,label) in enumerate(loader):
        real_data = real_data.view(-1,784).to(device)
        batch_size = real_data.shape[0]

        noise = torch.randn(batch_size,z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real_data).view(-1)
        loss_disc_real = criterion(disc_real,torch.ones_like(disc_real))

        disc_fake = disc(fake).view(-1)
        loss_disc_fake = criterion(disc_fake,torch.zeros_like(disc_fake))

        loss_disc = (loss_disc_real + loss_disc_fake)/2
        disc.zero_grad()
        loss_disc.backward(retain_graph = True)
        opt_disc.step()


        #generator
        output = disc(fake).view(-1)
        loss_gen = criterion(output,torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_dx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] \ "  
                f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1,1,28,28)
                data = real_data.reshape(-1,1,28,28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real_data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step = step
                )

                writer_real.add_image(
                    "Mnist Fake Images", img_grid_real, global_step = step
                )

                step += 1