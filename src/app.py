import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import io
import os
import zipfile

from skimage.measure import block_reduce
from sklearn.utils import Bunch
from PIL import Image


# Lê uma imagem
# def read_img(image_bytes, gray_scale=False):
#     image_src = cv2.imread(image_bytes)
#     if gray_scale:
#         image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
#     else:
#         image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
#     return image_src


# Converte os pixels da imagem para preto e/ou branco,
# 	de acordo com um threshold informado
def binarize_pixel(image_matrix, thresh):
    white = 255
    black = 0

    initial_conv = np.where((image_matrix <= thresh), image_matrix, white)
    final_conv = np.where((initial_conv > thresh), initial_conv, black)

    return final_conv


# Binariza uma imagem, ou seja, converte para preto e branco
# def binarize_img(image_src, threshold=128, with_plot=False, gray_scale=False):
#     # image_src = load_image(image_data)
#     if not gray_scale:
#         cmap_val = None
#         r_comp, g_comp, b_comp = image_src[:, :,
#                                            0], image_src[:, :, 1], image_src[:, :, 2]

#         r_b = binarize_pixel(image_matrix=r_comp, thresh=threshold)
#         g_b = binarize_pixel(image_matrix=g_comp, thresh=threshold)
#         b_b = binarize_pixel(image_matrix=b_comp, thresh=threshold)

#         image_b = np.dstack(tup=(r_b, g_b, b_b))
#     else:
#         cmap_val = 'gray'
#         image_b = binarize_pixel(image_matrix=image_src, thresh=threshold)

#     if with_plot:
#         fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))

#         ax1.axis("off")
#         ax1.title.set_text('Original')

#         ax2.axis("off")
#         ax2.title.set_text("Convertido")

#         ax1.imshow(image_src, cmap=cmap_val)
#         ax2.imshow(image_b, cmap=cmap_val)

#         # st.image(image_src)
#         st.image(image_b)
#         st.write(image_b)

#     return image_b

def reduce_blocks(image_src):
    return block_reduce(image_src/255/64, block_size=(8, 8)).round(2)


def binarize(image_src):
    image_b = binarize_pixel(image_matrix=image_src, thresh=128)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(2, 4))

    ax1.axis("off")
    ax1.title.set_text('Original')

    ax2.axis("off")
    ax2.title.set_text("Convertido")

    ax1.imshow(image_src, cmap='gray')
    ax2.imshow(image_b, cmap='gray')

    return image_b


# Carrega as imagens da pasta assets/images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        image_src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            images.append(image_src)
    return images


def train_images(images):
    # Convertendo imagens para preto e branco
    images_b = []
    for image in images:
        images_b.append(binarize(image))
    st.write('Originais (64x64)')
    st.image(images)
    st.write('Convertidos (64x64)')
    st.image(images_b)

    # Reduz os blocos das imagens (64x64 >> 8x8)
    images_r = []
    for image_b in images_b:
        images_r.append(reduce_blocks(image_b))

    st.write('Reduzidas (8x8)')
    st.image(images_r)
    # st.write(images_r)

    # Cria a estrutura de dados 'Bunch'
    b = Bunch()

    return


# Método principal
def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.title('Trabalho 2 - Inteligência Artificial')
    st.header('Reconhecimento de caracteres')
    
    st.info('As imagens devem estar disponíveis na pasta "assets/images" com o formato ".png" e tamanho 64x64, numeradas de 0 a n.\n\nEx.: 0.png, 1.png, ..., n.png')

    # Cria o menu lateral
    menu = st.sidebar

    debug = menu.checkbox('Modo de Debug')

    # Lê todos os arquivos de imagem dentro da pasta asseto
    images = load_images_from_folder('./assets/images')
    # st.image(images)

    train_images(images)
    
    # A = 0
    # B = 1
    # C = 2
    # D = 3
    # E = 4
    # F = 5

    st.info('Tabela de mapeamento')
    map_data = pd.read_csv('./assets/map.csv')
    st.write(map_data)
 
    # target = np.array([])
    # target_names = pd.DataFrame()

    # menu.write('Diego')


# Método Principal para rodar o programa
if __name__ == "__main__":
    main()
