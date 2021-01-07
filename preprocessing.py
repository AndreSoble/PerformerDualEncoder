import gzip
import os
import shutil
import tarfile
import traceback
from concurrent.futures.thread import ThreadPoolExecutor
from pprint import pprint
from random import shuffle

import requests
from tqdm import tqdm


def download_and_extract(url="https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-v1.0.tar.gz",
                         path="./storage"):
    if not os.path.isdir(path):
        os.mkdir(path)
    if os.path.isdir(path + "/opus-100-corpus/v1.0"):
        print("Already downloaded the data")
        return True
    try:
        print("Downloading data...")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(path + "/" + 'opus-100-corpus-v1.0.tar.gz', 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        # open('opus-100-corpus-v1.0.tar.gz', 'wb').write(response.content)
        print("Extracting...")
        with gzip.open(path + "/" + 'opus-100-corpus-v1.0.tar.gz', 'rb') as f_in:
            with open(path + "/" + 'opus-100-corpus-v1.0.tar', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        tf = tarfile.open(path + "/" + "opus-100-corpus-v1.0.tar")
        tf.extractall(path)
        return True
    except Exception:
        traceback.print_exc()
        return False


class SentencePair:
    def __init__(self, s1, s2):
        self.src = s1
        self.tgt = s2

    def get_target(self):
        return self.tgt

    def get_source(self):
        return self.src

    def __str__(self):
        return self.src + " -- " + self.tgt


class Corpus:
    def __init__(self, downsampled=False, downsampled_count=1000):
        self.dev = list()
        self.eval = list()
        self.train = list()
        self.pbar = None
        self.downsampled = downsampled
        self.downsampled_count = downsampled_count

    def get_eval(self):
        return self.eval

    def get_dev(self):
        return self.dev

    def get_train(self, shuffled=False):
        if shuffled:
            shuffle(self.train)
        return self.train

    def get_source_target_filename(self, f1, f2):
        src, tgt = "", ""
        if ".en" in f1:
            tgt = f1
            src = f2
        else:
            tgt = f2
            src = f1
        return src, tgt

    def add_2_corpus(self, iterable, mode=""):
        iterable = iterable if not self.downsampled else iterable[0:self.downsampled_count]
        for s1, s2 in iterable:
            if len(s1) < 2 or len(s2) < 2:
                continue
            if len(s1.split(" ")) < 2 or len(s2.split(" ")) < 2:
                continue
            if len(s1.split(" ")) > 128 or len(s2.split(" ")) > 128:
                continue
            if mode == "dev":
                self.dev.append(SentencePair(s1.replace("\n", ""), s2.replace("\n", "")))
            elif mode == "train":
                self.train.append(SentencePair(s1.replace("\n", ""), s2.replace("\n", "")))
            elif mode == "test":
                self.eval.append(SentencePair(s1.replace("\n", ""), s2.replace("\n", "")))

    def load_parallel(self, lang_folder_path):
        # Load all parallel sentences from the path
        # lang_folder_path = data_folder_path + "/" + lang_folder
        dev_files_paths = list()
        test_files_paths = list()
        train_files_paths = list()
        special_case = False
        for file in os.listdir(lang_folder_path):

            if "dev" in file:
                dev_files_paths.append(lang_folder_path + "/" + file)
            elif "test" in file:
                test_files_paths.append(lang_folder_path + "/" + file)
            elif "train" in file:
                train_files_paths.append(lang_folder_path + "/" + file)
            if ".es" in file:
                special_case = True
        if len(dev_files_paths) > 1:
            dev_src_path, dev_tgt_path = self.get_source_target_filename(dev_files_paths[0], dev_files_paths[1])
        if len(test_files_paths) > 1:
            test_src_path, test_tgt_path = self.get_source_target_filename(test_files_paths[0], test_files_paths[1])
        if len(train_files_paths) > 1:
            train_src_path, train_tgt_path = self.get_source_target_filename(train_files_paths[0],
                                                                             train_files_paths[1])

        self.add_2_corpus(
            zip(open(dev_src_path, encoding="utf-8").readlines(), open(dev_tgt_path, encoding="utf-8").readlines()),
            mode="dev")
        self.add_2_corpus(zip(open(test_src_path, encoding="utf-8").readlines(),
                              open(test_tgt_path, encoding="utf-8").readlines()), mode="test")
        self.add_2_corpus(zip(open(train_src_path, encoding="utf-8").readlines(),
                              open(train_tgt_path, encoding="utf-8").readlines()), mode="train")

        if special_case:
            special_case = False
            self.add_2_corpus(zip(open(dev_tgt_path, encoding="utf-8").readlines(),
                                  open(dev_tgt_path, encoding="utf-8").readlines()), mode="dev")
            self.add_2_corpus(zip(open(test_tgt_path, encoding="utf-8").readlines(),
                                  open(test_tgt_path, encoding="utf-8").readlines()), mode="test")
            self.add_2_corpus(zip(open(train_tgt_path, encoding="utf-8").readlines(),
                                  open(train_tgt_path, encoding="utf-8").readlines()), mode="train")
        self.pbar.update(1)

    def load_corpus(self, path="./storage", debug=False):
        assert os.path.isdir(path + "/opus-100-corpus/v1.0/supervised")  # data not found
        data_folder_path = path + "/opus-100-corpus/v1.0/supervised"
        print("Loading corpus...")
        print("")
        threadpool_input = [data_folder_path + "/" + lang_folder for lang_folder in os.listdir(data_folder_path)]
        if debug:
            threadpool_input = [t for t in threadpool_input if "en-es" in t]
            print("Debug - Loading only one language -", threadpool_input)
        with tqdm(total=len(threadpool_input)) as self.pbar:
            with ThreadPoolExecutor(max_workers=6) as executor:
                executor.map(self.load_parallel, threadpool_input)

    def get_stats(self, path="./storage"):
        print("Computing stats...")
        assert os.path.isdir(path + "/opus-100-corpus/v1.0/supervised")
        data_folder_path = path + "/opus-100-corpus/v1.0/supervised"
        src_tgt_langs = dict()
        general_counter = 0
        for folder in tqdm(os.listdir(data_folder_path)):
            src_tgt_langs[folder] = 0
            for file in os.listdir(data_folder_path + "/" + folder):
                if "train" not in file and ".en" not in file:
                    continue
                file = open(data_folder_path + "/" + folder + "/" + file)
                for s1 in file.readlines():
                    if len(s1) < 2:
                        continue
                    if len(s1.split(" ")) < 2:
                        continue
                    if len(s1.split(" ")) > 300:
                        continue
                    src_tgt_langs[folder] += 1
                    general_counter += 1
                file.close()
        pprint(src_tgt_langs)
        print(f"Amount of languages is {len(list(src_tgt_langs.keys()))}")
        print(f"Not perfect precise amount of training data is {general_counter}")


if __name__ == "__main__":
    # assert download_and_extract(path=os.environ.get("DATA", "/mnt/mlag/andre_soblechero/storage"))
    corpus = Corpus()
    corpus.get_stats("/mnt/d/backup/IdeaProjects/Transformer Autoencoder/storage")
    # corpus.load_corpus(debug=False, path=os.environ.get("DATA_DIR", "./storage"))
