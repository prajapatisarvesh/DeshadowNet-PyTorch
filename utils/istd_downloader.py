import gdown
import zipfile
import os
import patoolib


if __name__ == '__main__':
    file_name = 'istd.rar'
    download_link = 'https://drive.google.com/u/1/uc?id=1I0qw-65KBA6np8vIZzO6oeiOvcDBttAY'
    os.chdir('..')
    if not os.path.exists('data'):
        os.mkdir('data')
    os.chdir('data')
    if not os.path.exists('ISTD_Dataset') and not os.path.exists(file_name):
        print("[+] ISTD Data Downloader")
        print("[+] Downloading ISTD.rar")
        gdown.download(url=download_link, output=file_name, quiet=False)
        patoolib.extract_archive(file_name, outdir=os.getcwd())
        print('[+] Cleanup')
        os.remove(file_name)
    os.chdir('ISTD_Dataset')
    parent_dir = os.getcwd()
    main_folders = ['train', 'test']
    for folder in main_folders:
        child_folders = os.listdir(os.path.join(parent_dir, folder))
        shadow_images = []
        shadow_masks = []
        shadow_free_images = []
        cdir = os.path.join(os.getcwd(), folder)
        for child in child_folders:
            if 'A' in child:
                shadow_images = os.listdir(os.path.join(cdir, child))
                shadow_images = [f'{os.path.join(cdir, child, a)}' for a in shadow_images]
            if 'B' in child:
                shadow_masks = os.listdir(os.path.join(cdir, child))
                shadow_masks = [f'{os.path.join(cdir, child, a)}' for a in shadow_masks]
            if 'C' in child:
                shadow_free_images = os.listdir(os.path.join(cdir, child))
                shadow_free_images = [f'{os.path.join(cdir, child, a)}' for a in shadow_free_images]
        shadow_images.sort()
        shadow_masks.sort()
        shadow_free_images.sort()
        with open(f'{folder}.csv', 'w') as f:
            f.write('shadow_image,shadow_mask,shadow_free_image\n')
            for a,b,c in zip(shadow_images, shadow_masks, shadow_free_images):
                f.write(f'{a},{b},{c}\n')