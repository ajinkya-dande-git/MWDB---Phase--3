import json
import os
from os import path
import pickle
import scipy.spatial
import numpy as np

from Main.helper import get_rand_vec, determine_pnt_side, ret_files_without_ext, ret_n_size_bin_strings, progress, find_distance_2_vectors
import Main.config as config


hash_rep = []
database = []


# Generates a Hash Table Representation of random vectors to be projected on.
def gen_hash_rep(l, k, vec_dims):
    # hash_rep = []
    global hash_rep
    global database
    for i in range(l):
        hash_layer = []
        for j in range(k):
            hash_layer.append(get_rand_vec(vec_dims))
        hash_rep.append(hash_layer)
    # return hash_rep

# Generate img buckets for specific layer
# def img_index(hash_rep, feat_desc, layer_ptr):
def img_index(feat_desc, layer_ptr):
    global hash_rep
    global database
    bin_str = ''
    for i in range(len(hash_rep[layer_ptr])):
        dot_pdt = determine_pnt_side(hash_rep[layer_ptr][i], feat_desc)
        if (dot_pdt < 0):
            bin_str += '0'
        else:
            bin_str += '1'
    return bin_str


# Index Imgs
# def index_imgs(hash_rep):
def index_imgs():
    global database
    global hash_rep
    database = []
    json_files = os.listdir(config.FEATURES_FOLDER)
    json_files = ret_files_without_ext(json_files)

    ctr = 0
    total_iter = len(hash_rep) * len(json_files)
    for i in range(len(hash_rep)):
        layer_index = {}

        for img in json_files:
            with open(path.join(config.FEATURES_FOLDER, img + '.json'), 'r', encoding='utf-8') as f:
                feat_desc = json.load(f)
            img_bucket = img_index(feat_desc[img], i)

            if (img_bucket not in layer_index):
                layer_index[img_bucket] = [img]
            else:
                layer_index[img_bucket].append(img)
            ctr += 1
            progress(ctr, total_iter)

        database.append(layer_index)

    return database


# Driver function
def lsh_index_structure(l, k):
    global hash_rep
    global database
    with open(path.join(config.FEATURES_FOLDER, 'Hand_0000002.jpg.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    vec_dims = len(data['Hand_0000002.jpg'])   # Size of HOG Feature descriptor stored.
    gen_hash_rep(l, k, vec_dims)
    # print(hash_rep)
    database = index_imgs()
    return database


def visualize_for_lsh(sorted_imgs, l, k, t, query_img):
    global hash_rep
    global database
    s = "<style>" \
        ".images { width:160px;height:120px;float:left;margin:20px;}" \
        "img{width:160px;height:120px;}" \
        "</style>"
    s = s + "<h2 style='text-align:center;margin-top:60px;'> List of similar imgs </h2><div class='container'>"

    for img in sorted_imgs[:t]:
        news = "<div class='images'>"
        news = news + "<img src='"
        news = news + path.join(config.FULL_IMAGESET_FOLDER, img[0])
        news = news + "'><div style='text-align:center;'> Image: <span style='font-weight:bold;'>" + img[0] + "</span></div>"
        news = news + "</div>"
        s = s + news

    s = s + "</div>"
    f = open(path.join(config.DATABASE_FOLDER, "task5_" + query_img + "_l" + str(l) + "_k" + str(k) + "_t" + str(t) + ".html"), "w")
    f.write(s)
    f.close()


def store_results_as_json(sorted_imgs, l, k, t, query_img):
    global hash_rep
    global database
    results_dict = dict()
    for img in sorted_imgs[:t]:
        with open(path.join(config.FEATURES_FOLDER, img[0] + '.json'), 'r', encoding='utf-8') as f:
            img_fd = json.load(f)[img[0]]
        results_dict[img[0]] = img_fd
    with open(path.join(config.DATABASE_FOLDER, "task5_" + query_img + "_l" + str(l) + "_k" + str(k) + "_t" + str(t) + ".json"), "w") as f:
        json.dump(results_dict, f)


def store_pickles(l, k):
    global hash_rep
    global database
    filehandler = open(path.join(config.DATABASE_FOLDER, "task5_hashRep_l"+str(l)+"_k"+str(k)), "wb")
    pickle.dump(hash_rep, filehandler)
    filehandler = open(path.join(config.DATABASE_FOLDER, "task5_indexedImgs_l"+str(l)+"_k"+str(k)), "wb")
    pickle.dump(database, filehandler)


def visualize_for_task6(sorted_imgs, l, k, t):
    global hash_rep
    global database

    s = "<style>" \
        ".images { width:160px;height:120px;float:left;margin:20px;}" \
        "img{width:160px;height:120px;}" \
        "</style>"
    s = s + "<h2 style='text-align:center;margin-top:60px;'> List of similar imgs </h2><div class='container'>"

    for img in sorted_imgs[:t]:
        news = "<div class='images'>"
        news = news + "<img src='"
        news = news + path.join(config.FULL_IMAGESET_FOLDER, img[0])
        news = news + "'><div style='text-align:center;'> Image: <span style='font-weight:bold;'>" + img[
            0] + "</span></div>"
        news = news + "</div>"
        s = s + news

    s = s + "</div>"
    f = open(path.join(config.DATABASE_FOLDER, "task5_" + "unseenImg" + "_l" + str(l) + "_k" + str(k) + "_t" + str(t) + ".html"), "w")
    f.write(s)
    f.close()


def store_results_for_task6(sorted_imgs, l, k, t):
    global hash_rep
    global database
    results_dict = dict()
    for img in sorted_imgs[:t]:
        with open(path.join(config.FEATURES_FOLDER, img[0] + '.json'), 'r', encoding='utf-8') as f:
            img_fd = json.load(f)[img[0]]
        results_dict[img[0]] = img_fd
    with open(path.join(config.DATABASE_FOLDER, "task5_" + "unseenImg" + "_l" + str(l) + "_k" + str(k) + "_t" + str(t) + ".json"), "w") as f:
        json.dump(results_dict, f)


def task_6_inp(query_img_fd, l, k, t):
    global hash_rep
    global database
    with open(path.join(config.DATABASE_FOLDER, "task5_hashRep_l"+str(l)+"_k"+str(k)), "rb") as f:
        hash_rep = pickle.load(f)
    with open(path.join(config.DATABASE_FOLDER, "task5_indexedImgs_l"+str(l)+"_k"+str(k)), "rb") as f:
        database = pickle.load(f)

    query_rep = []
    for i in range(len(hash_rep)):
        img_bucket = img_index(query_img_fd, i)
        query_rep.append(img_bucket)
    # print(query_rep)

    # Get list of similar imgs
    similar_imgs = []
    bits_to_ignore = 0
    while (1):

        bins_to_consider = set()
        similar_imgs = []
        if (bits_to_ignore > 0):
            arr = [None] * bits_to_ignore
            ret_n_size_bin_strings(bits_to_ignore, arr, 0, bins_to_consider)

        for layer_ptr in range(len(database)):
            curr_layer_bit_rep = query_rep[layer_ptr]
            if bits_to_ignore > 0:
                for bin in bins_to_consider:
                    key = curr_layer_bit_rep[: -1 * bits_to_ignore] + bin
                    if key in database[layer_ptr]:
                        similar_imgs.extend(database[layer_ptr][key])
            else:
                if curr_layer_bit_rep in database[layer_ptr]:
                    similar_imgs.extend(database[layer_ptr][curr_layer_bit_rep])

        if (len(similar_imgs) >= t or bits_to_ignore == len(query_rep[0])):
            break  # Also look to break when all the buckets have been scanned.
        else:
            # Reduce the bit size
            bits_to_ignore += 1

    total_imgs = len(similar_imgs)
    unique_similar_imgs = set(similar_imgs)
    unique_imgs = len(unique_similar_imgs)
    print('Number of unique images: ', unique_imgs)
    print('Overall number of images considered: ', total_imgs)

    # Rank all unique images.
    euclid_dist = dict()
    query_fd = query_img_fd
    for img in unique_similar_imgs:
        # print(img)
        with open(path.join(config.FEATURES_FOLDER, img + '.json'), 'r', encoding='utf-8') as f:
            img_fd = json.load(f)
        euclid_dist[img] = find_distance_2_vectors(np.asarray(query_fd), np.asarray(img_fd[img]))

    sorted_imgs = sorted(euclid_dist.items(), key=lambda kv: kv[1])
    # print(sorted_imgs[:t])

    visualize_for_task6(sorted_imgs, l, k, t)
    store_results_for_task6(sorted_imgs, l, k, t)


def startTask5():
    global hash_rep
    global database
    l = int(input('Enter layer: '))
    k = int(input('Enter hashes per layer: '))

    database = lsh_index_structure(l, k)

    query_img = input('Enter the Query Image: ')
    t = int(input('Most similar images (t): '))


    # Compute hash rep for query img.
    with open(path.join(config.FEATURES_FOLDER, query_img + '.json'), 'r', encoding='utf-8') as f:
        query_feat_desc = json.load(f)

    query_rep = []
    for i in range(len(hash_rep)):
        img_bucket = img_index(query_feat_desc[query_img], i)
        query_rep.append(img_bucket)
    # print(query_rep)

    # Get list of similar imgs
    similar_imgs = []
    bits_to_ignore = 0
    while (1):

        bins_to_consider = set()
        if (bits_to_ignore > 0):
            arr = [None] * bits_to_ignore
            ret_n_size_bin_strings(bits_to_ignore, arr, 0, bins_to_consider)

        for layer_ptr in range(len(database)):
            curr_layer_bit_rep = query_rep[layer_ptr]
            if bits_to_ignore > 0:
                for bin in bins_to_consider:
                    key = curr_layer_bit_rep[: -1 * bits_to_ignore] + bin
                    if key in database[layer_ptr]:
                        similar_imgs.extend(database[layer_ptr][key])
            else:
                if curr_layer_bit_rep in database[layer_ptr]:
                    similar_imgs.extend(database[layer_ptr][curr_layer_bit_rep])

        if (len(similar_imgs) >= t or bits_to_ignore == len(query_rep[0])):
            break  # Also look to break when all the buckets have been scanned.
        else:
            # Reduce the bit size
            bits_to_ignore += 1
            del similar_imgs[:]

    total_imgs = len(similar_imgs)
    unique_similar_imgs = set(similar_imgs)
    unique_imgs = len(unique_similar_imgs)
    print('Number of unique images: ', unique_imgs)
    print('Overall number of images considered: ', total_imgs)

    # Rank all unique images.
    euclid_dist = dict()
    query_fd = query_feat_desc[query_img]
    for img in unique_similar_imgs:
        # print(img)
        with open(path.join(config.FEATURES_FOLDER, img + '.json'), 'r', encoding='utf-8') as f:
            img_fd = json.load(f)
        euclid_dist[img] = find_distance_2_vectors(np.asarray(query_fd), np.asarray(img_fd[img]))

    sorted_imgs = sorted(euclid_dist.items(), key=lambda kv: kv[1])
    # print(sorted_imgs[:t])

    visualize_for_lsh(sorted_imgs, l, k, t, query_img)
    store_results_as_json(sorted_imgs, l, k, t, query_img)
    store_pickles(l, k)



if __name__ == '__main__':
    startTask5()