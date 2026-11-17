"""Read-only minimind resource admission and parallel-wave sampling."""
import json
import re
import subprocess
import threading
import time

def read_resources(groups=()):
    pressure = subprocess.check_output(['/usr/bin/memory_pressure','-Q'],text=True,timeout=6)
    vm = subprocess.check_output(['/usr/bin/vm_stat'],text=True,timeout=6)
    processes = subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,pcpu='],text=True,timeout=6)
    return parse_resources(pressure,vm,processes,groups)

def parse_resources(pressure,vm,processes,groups):
    available = int(re.search(r'System-wide memory free percentage:\s*(\d+)%',pressure)[1])
    page_size = int(re.search(r'page size of (\d+) bytes',vm)[1])
    swapouts = int(re.search(r'^Swapouts:\s*(\d+)\.',vm,re.M)[1])
    owned = {int(g):dict(rssKiB=0,cpuLifetimePercent=0.,pids=[]) for g in groups}
    for line in processes.splitlines():
        pid,group,rss,cpu=line.split();group=int(group)
        if group in owned:
            owned[group]['rssKiB']+=int(rss);owned[group]['cpuLifetimePercent']+=float(cpu);owned[group]['pids'].append(int(pid))
    return dict(availablePercent=available,pageSize=page_size,swapouts=swapouts,ownedGroups=owned)

def annotate_sample(previous,sample,now,streak):
    sample['monotonic']=now
    if previous is not None:
        delta=sample['swapouts']-previous['swapouts'];elapsed=now-previous['monotonic']
        if delta<0 or elapsed<=0 or sample['pageSize']!=previous['pageSize']:raise ValueError('Resource counter reset')
        sample['elapsedSeconds']=elapsed;sample['swapoutPagesPer2Seconds']=delta*2/elapsed
        streak=streak+1 if sample['swapoutPagesPer2Seconds']>128 else 0
    sample['swapoutBreachStreak']=streak
    return streak,('Memory/paging resource guard breached' if sample['availablePercent']<20 or streak>=3 else None)

class Monitor:
    def __init__(self,path):
        self.path=path;self.groups=set();self.lock=threading.Lock();self.done=threading.Event()
        self.error=None;self.samples=0;self.thread=None
    def register(self,pid):
        with self.lock:self.groups.add(pid)
    def start(self):
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()
    def _loop(self):
        previous=None;streak=0;deadline=time.monotonic()
        try:
            with self.path.open('x') as output:
                while not self.done.is_set():
                    with self.lock:groups=tuple(self.groups)
                    sample=read_resources(groups);now=time.monotonic()
                    streak,error=annotate_sample(previous,sample,now,streak)
                    output.write(json.dumps(sample)+'\n');output.flush();self.samples+=1
                    if error:self.error=error
                    previous=sample;deadline+=2
                    self.done.wait(max(0,deadline-time.monotonic()))
        except BaseException as error:self.error='Resource sampling failed: '+repr(error)
    def close(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=20)
            if self.thread.is_alive():self.error='Resource monitor did not finish'
        if self.samples<2:self.error=self.error or 'Insufficient wave resource samples'
