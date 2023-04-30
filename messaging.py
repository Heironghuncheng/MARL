# coding=utf-8

from httpx import post
import time
from tensorflow import Tensor


def default(obj) -> str:
    if isinstance(obj, (dict,)):
        return "".join([str(items[0]) + ": " + default(items[1]) for items in obj.items()])
    elif isinstance(obj, (tuple,)):
        return "".join([default(items) for items in obj])
    elif isinstance(obj, (Tensor,)):
        return str(obj.numpy().tolist()) + "\n"
    else:
        return str(obj) + "\n"


class Messaging(object):
    def __init__(self, version: str = "single", token: int = 219855854527111798587013771002803041281):
        self.token = token
        self.version = version

    def auth(self) -> str:
        code = ""
        while self.token > 0:
            code += chr(self.token % 100 + 66)
            self.token //= 100
        return code

    def messaging(self, mes_type: str, spot: dict, remark: str):
        msg = {"VERSION": self.version, "TIME": str(time.ctime(time.time())), "TYPE": mes_type, "SPOT": spot,
               "REMARK": remark}
        msg = default(msg)
        data = {
            'auth': self.auth(),
            'msg': msg,
            'id': 2978103904,
            'type': 'private'
        }
        res = post('https://woc.8am.run/webapi/custom-msg', json=data, timeout=1000)
        print(res)


if __name__ == "__main__":
    mes = Messaging()
    mes.messaging("test", {"a": "b", "c": "d", "e": {"f": "g"}}, "none")
