import gdown

# define path to downloads and the outputs
SIMCLR_URL = "https://drive.google.com/drive/folders/1HkC1tWVU0Q3weIQup5GQjJYeiADS_VwV?usp=sharing"
INCEPTION_URL = "https://drive.google.com/drive/folders/1B__QjvTdzk-wJoRPCXgHYcncwT5-kDjm?usp=sharing"

INCEPTION_OUTPUT = "./model/saved/inceptionv3_net"
SIMCLR_OUTPUT = "./model/saved/simclrv2_net"

if __name__ == "__main__":
    # download local files and update the models
    print("Downloading models ...")
    gdown.download_folder(SIMCLR_URL, output=SIMCLR_OUTPUT, quiet=False, use_cookies=False)
    gdown.download_folder(INCEPTION_URL, output=INCEPTION_OUTPUT, quiet=False, use_cookies=False)