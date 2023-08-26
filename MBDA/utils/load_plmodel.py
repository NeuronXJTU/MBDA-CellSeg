def load_stat_dict(state_dict):
    new_stat_dict = {}
    for k,v in state_dict.items():
        if k.split('.')[0] == 'model':
            new_stat_dict[k[6:]] = v
    return new_stat_dict