"""entrypoint for tls-gatherer module"""
import sys
import task_scheduler
import time

sys.setrecursionlimit(10000)

QUEUE = {"https://www.google.de": ["google", "search_engine"],
         "https://www.youtube.com": ["google", "streaming"],
         "https://www.facebook.com": ["social media"], "https://www.yahoo.com": ["newspage"]}

try:
    # C_MANAGER = task_scheduler.Supervisor()
    # C_MANAGER.initiate()
    print("Start container")
except Exception as e:
    print(e)
    print("SOME FAILURE OCCURRED")
time.sleep(60)
