import hashlib
import pickle

file = 'D:\code\code_xwd\Durable-Federated-Learning-Backdoor\FL_Backdoor_CV\data\poison_cifar10\data_batch_1'

md5_hash = hashlib.md5()
with open(file,"rb") as f:
    # Read and update hash in chunks of 4K
    for byte_block in iter(lambda: f.read(4096),b""):
        md5_hash.update(byte_block)
    print(md5_hash.hexdigest())