

def format_time(delta_time):
    m, s = divmod(delta_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    dhms = [int(d), int(h), int(m), s]
    str_dhms = ['{:1d} day, ', '{:0>2d} hour, ', '{:0>2d} min, ', '{:0>5.2f} sec']
    for i in range(len(dhms)):
        if dhms[i] > 0:
            return ''.join(str_dhms[i:]).format(*dhms[i:])
    return '{:0>5.2f} sec'.format(0)
