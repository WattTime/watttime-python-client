import sched
from datetime import datetime
import time

scheduler = sched.scheduler(timefunc=time.time)

requery_interval = 5
sleep_seconds = 5*60

def main():
    """Do the main thing"""
    time.sleep(sleep_seconds)
    print('Done main')

def other():
    """Do the other thing"""
    time.sleep(2)
    print('Done other')

scheduler.enter(3, 2, main) # delay, priority, action, argument=()
scheduler.enter(2, 1, other)

def executing_at_message(event):
    date = datetime.fromtimestamp(event.time)
    print(f"{date.hour}:{date.minute}:{date.second} {event.action.__doc__}")

for event in scheduler.queue:
    executing_at_message(event)