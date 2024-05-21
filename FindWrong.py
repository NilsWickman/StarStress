import os
import pickle
from pickle import load, loads
from zstd import decompress
from util import get_names

def list_files(directory):
    """
    List all files in a directory and its subdirectories.
    """
    # Iterate over all files and directories in the given directory

    total = 0
    counter = 0
    seq_length = 32
    file_counter = 0
    replay_counter = 0
    for root, dirs, files in os.walk(directory):
        if counter == 0:
            counter += 1
            continue
        counter += 1
        names = get_names(root)
        rep_count = (len(names) - 1)
        multi_load = seq_length // 32
        for i in range(0, rep_count, multi_load):
                if i < rep_count:
                    with open(os.path.join(root, str(i)),
                              "rb") as f:
                        try:
                            loaded = loads(decompress(load(f)))
                        except EOFError:
                            print(counter, "Input Error for: ", f.name)
                            print("removing")
                            for file in files:
                                os.remove(str(root) + "/" + file)
                            os.rmdir(root)
                            break
                        except pickle.UnpicklingError:
                            print(counter, "Input Error for: ", f.name)
                            print("removing")
                            for file in files:
                                os.remove(str(root) + "/" + file)
                            os.rmdir(root)
                            break
        #if len(files) < 20:
        #    counter += 1
        #    for file in files:
        #        print(file, root)
    counter -= 1
    print("Amount of files: ", file_counter)
    print("Courupted Replays: ", replay_counter)
    #print("Files if conditional removed: ", (total - counter))
    #print("Average files per replay: ", (counter/len(files)))

# Example usage:
#directory_path = "replays/training_replays/"
#list_files(directory_path)