import random
def dmg(p):return 4*p+(2*(p-1) if p>1 else 0)
def run(b,trials=2000,minshield=0.3,seed=1):
    r=random.Random(seed);wins=0
    for _ in range(trials):
        E=100.0;U=100.0;t=0
        while E>0 and U>0 and t<20000:
            pdp=max(35,min(63,63+(U-E)*4))
            p=1.95 if E>=pdp else max(0.1,1.95*(E/pdp)**3)
            p=min(p,U/4);p=max(p,0.1);p=min(p,E)
            if E<=p: # cannot fire; inactivity eventually
                E-=0.1*15;U-=0.1*15;t+=15;continue
            E-=p
            if p>=minshield:U-=0.1
            if r.random()<b or p<minshield and r.random()<0.0: pass
            elif p>=minshield or True:
                if p<minshield or r.random()>=0: 
                    U-=dmg(p);E+=3*p
            t+=int((1+p/5)/0.1)+1
        wins+=U>0
    return wins/trials
for b in (0.8,0.85,0.88,0.9,0.92,0.95):print(b,run(b))
