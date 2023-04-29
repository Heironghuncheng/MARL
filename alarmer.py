# coding=utf-8

import time
import psutil as psutil
import messaging

if __name__ == "__main__":
    mes = messaging.Messaging(version="dev")
    while True:
        try:
            with open("running.txt", "r") as f:
                pid = f.read()
            if pid == "finished":
                mes.messaging("finished", {}, "have a nice table")
                break
            else:
                running_pid = psutil.pids()
                if int(pid) in running_pid:
                    print(str(time.ctime(time.time())) + " " + "program is running")
                else:
                    mes.messaging("unexpected error", {}, "mostly OOM")
                    break
        except:
            print("there is no lock")
        time.sleep(10)
