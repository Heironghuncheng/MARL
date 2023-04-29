# coding=utf-8

from httpx import post
import time
import json


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
        msg = {"version": self.version, "time": str(time.ctime(time.time())), "type": mes_type, "spot": spot,
               "remark": remark}
        data = {
            'auth': self.auth(),
            'msg': json.dumps(msg),
            'id': 2978103904,
            'type': 'private'
        }
        res = post('https://woc.8am.run/webapi/custom-msg', json=data, timeout=1000)
        print(res)





