import os

def install():
    from stable_pretraining.optim.lr_scheduler import DEFAULT_SCHEDULER_FACTORIES
    value=os.environ.get('WARMUP_STEPS','official')
    if value=='official':return
    count=int(value);assert count in [0,8,16,64]
    original=DEFAULT_SCHEDULER_FACTORIES['LinearWarmupCosineAnnealingLR']
    def factory(module,optimizer):
        params=original(module,optimizer)
        assert set(params)=={'warmup_steps','max_steps','warmup_start_lr','eta_min'}
        params['warmup_steps']=count
        return params
    DEFAULT_SCHEDULER_FACTORIES['LinearWarmupCosineAnnealingLR']=factory
