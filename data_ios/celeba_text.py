#Author: Yahui Liu <yahui.liu@unitn.it>

import random
import numpy as np

selected_attrs = {
    0: 'black',  # Black_Hair,
    1: 'blond',  # Blond_Hair,
    2: 'brown',  # Brown_Hair,
    3: 'male',   # Gender: 1, Male; 0, Female
    4: 'smile',  # Smile: 1, Smile; 0, Unsmiling
    5: 'young',  # Age: 1, Young; 0, Old
    6: 'eyeglasses', # Eyeglasses: 1, with; 0, without
    7: 'beard'   # No_beard: 1, No_beard; 0, Beard
}
reverse_selected_attrs = {v:k for k,v in selected_attrs.items()}
gender_index = reverse_selected_attrs['male']

color_type = ['black', 'blond', 'brown']
# for all
change_actions = ['make', 'change', 'translate', 'modify']
# for female, smiling, young
reverse_actions = ['change', 'reverse', 'inverse']
# for young
increase_actions = ['increase', 'add']
decrease_actions = ['decrease', 'reduce']

male_words = ['boy', 'male', 'man', 'gentleman', 'sir']
female_words = ['female', 'woman', 'lady', 'miss', 'girl']
beard_words = ['beard', 'moustache', 'whiskers', 'beards']
glasses_words = ['glasses', 'eyeglasses', 'sunglasses']
smile_words = ['smile', 'smiling', 'happy', 'delighted', 'laugh']
unsmile_words = ['unsmiling', 'unhappy', 'serious', 'smileless', 'solemn', 'less smile', 'more serious']
young_words = ['young', 'younger']
old_words = ['old', 'older', 'big age']
glasses_add = ['wear', 'add', 'put on', 'with']
glasses_remove = ['remove', 'take off', 'without', 'no']
beard_add = ['wear', 'add', 'put on', 'with']
beard_remove = ['remove', 'take off', 'without', 'no']


def _get_gender(is_male):
    return random.choice(['his' if is_male else 'her', 'the'])

def do_nothing(text='', is_male=0):
    gender = _get_gender(is_male)
    if text == '':
        return random.choice([
            '',
            'do nothing', 
            'no changes',
            'do not change anything'
        ])

    return random.choice([
        '',
        'do nothing on {} {}'.format(gender, text),
        'do not change {} {}'.format(gender, text),
        'keep {} unchanged'.format(text),
        'keep {} {} unchanged'.format(gender, text)
    ])

def get_colors(nonzeros, use_shuffle=True):
    if len(nonzeros) == 0:
        return 'unknown'
    colors = []
    for i in range(len(nonzeros)):
        colors.append(color_type[nonzeros[i]])
    if use_shuffle:
        random.shuffle(colors)
    if len(nonzeros) < 3:
        color_txt = ' and '.join(colors)
    else:
        color_txt = ' , '.join(colors[:-1]) + ' and {}'.format(colors[-1])
    return color_txt

def edit_hair_color(src_lab, tgt_lab):
    src_slice = src_lab[:3]
    trg_slice = tgt_lab[:3]
    if np.sum(np.abs(src_slice-trg_slice)) == 0:
        return do_nothing('hair color', tgt_lab[gender_index])

    target_color  = np.nonzero(trg_slice)[0]
    trg_color_txt = get_colors(target_color)
    source_color  = np.nonzero(src_slice)[0]
    src_color_txt = get_colors(source_color)
    gender = _get_gender(tgt_lab[gender_index])
    color = random.choice(['color', 'colour'])
    txt = random.choice([
        '{} hair {} {}'.format(random.choice(change_actions), 
            color, trg_color_txt), 
        '{} {} {} hair {} {}'.format(random.choice(change_actions), 
            gender, src_color_txt, random.choice(['to', 'into']), trg_color_txt),
        '{} {} hair {} from {} {} {}'.format(random.choice(change_actions), 
            gender, color, src_color_txt, random.choice(['to', 'into']), trg_color_txt),
        '{} hair'.format(trg_color_txt),
        '{} hair {}'.format(trg_color_txt, color)
    ])
    return txt

def edit_gender(src, trg):
    src_gender = _get_gender(src)
    trg_gender = _get_gender(trg)

    if src-trg == 0:
        return random.choice([
            do_nothing('gender', src),
            '{} {} gender {} {}'.format(random.choice(change_actions), src_gender, 
                random.choice(['to', 'into']),
                random.choice(male_words) if trg else random.choice(female_words)),
            '{} the gender from {} {} {}'.format(random.choice(change_actions), 
                random.choice(male_words) if src else random.choice(female_words),
                random.choice(['to', 'into']),
                random.choice(male_words) if trg else random.choice(female_words))
        ])

    txt = random.choice([
        '{} {} gender'.format(random.choice(change_actions), src_gender),
        '{} {} gender {} {}'.format(random.choice(change_actions), src_gender, 
            random.choice(['to', 'into']),
            random.choice(male_words) if trg else random.choice(female_words)),
        '{} the gender from {} {} {}'.format(random.choice(change_actions), 
            random.choice(male_words) if src else random.choice(female_words),
            random.choice(['to', 'into']),
            random.choice(male_words) if trg else random.choice(female_words)),
        '{} the gender'.format(random.choice(reverse_actions)),
        '{} gender'.format(random.choice(reverse_actions)),
        '{} the {} to be a {}'.format(random.choice(change_actions), 
            random.choice(male_words+['face']) if src else random.choice(female_words+['face']),
            random.choice(male_words+['face']) if trg else random.choice(female_words+['face'])),
        '{}'.format(random.choice(male_words) if trg else random.choice(female_words))
    ])
    return txt

def edit_smiling(src, trg, is_male=0):
    gender1 = _get_gender(is_male)
    gender2 = random.choice(['him' if is_male else 'her', 'it'])

    if src-trg == 0:
        return random.choice([
            'keep {} face {}'.format(gender1, 
                random.choice(smile_words) if trg else random.choice(unsmile_words)),
            'keep {} {}'.format(gender2, 
                random.choice(smile_words) if trg else random.choice(unsmile_words)),
            do_nothing(random.choice(smile_words) if trg else random.choice(unsmile_words), 
                is_male)
        ])

    status = random.choice(beard_add) if trg else random.choice(beard_remove)
    txt = random.choice([
        random.choice(smile_words) if trg else random.choice(unsmile_words),
        '{} {} face {}'.format(random.choice(change_actions), gender1, 
            random.choice(smile_words) if trg else random.choice(unsmile_words)),
        '{} {} face to be {}'.format(random.choice(change_actions), gender1,
            random.choice(smile_words) if trg else random.choice(unsmile_words)),
        '{} {}'.format(status, random.choice(['smile', 'the smile'])),
        'smile' if trg else "do not smile"
    ])
    return txt

def edit_age(src, trg, is_male=0):
    if src-trg == 0:
        return do_nothing('age', is_male)
    
    gender1 = _get_gender(is_male)
    gender2 = random.choice(['him' if is_male else 'her', 'it'])
    src_age = random.choice(young_words) if src else random.choice(old_words)
    trg_age = random.choice(young_words) if trg else random.choice(old_words)
    
    txt = random.choice([
        trg_age, 
        '{} {} face {}'.format(random.choice(change_actions), gender1, trg_age),
        '{} {} {}'.format(random.choice(change_actions), gender2, trg_age),
        '{} {} face {} be {}'.format(random.choice(change_actions), gender1, 
            random.choice(['to', 'into']), trg_age),
        '{} {} to be {}'.format(random.choice(change_actions), gender2, trg_age),
        '{} age'.format(random.choice(increase_actions) if trg else random.choice(decrease_actions)),
        '{} {} age'.format(random.choice(reverse_actions), gender1),
        '{} {} age'.format(random.choice(increase_actions) if trg else random.choice(decrease_actions), 
            gender1)
    ])
    return txt

def edit_eyeglasses(src, trg, is_male=0):
    if src-trg == 0:
        return do_nothing('eyeglasses', is_male)
    gender1 = _get_gender(is_male)
    gender2 = random.choice(['him' if is_male else 'her', 'it'])

    status = random.choice(glasses_add) if trg else random.choice(glasses_remove)
    txt = random.choice([
        '{} {}'.format(status, random.choice(glasses_words)),
        '{} {} face {} {}'.format(random.choice(change_actions), gender1,
            status, random.choice(glasses_words)),
        '{} {} {} {}'.format(random.choice(change_actions), gender2,
            status, random.choice(glasses_words))
    ])
    return txt

def edit_beard(src, trg, is_male=0):
    if src-trg == 0:
        return do_nothing('beard', is_male)
    
    gender1 = _get_gender(is_male)
    gender2 = random.choice(['him' if is_male else 'her', 'it'])
    status = random.choice(beard_remove) if trg else random.choice(beard_add)
    txt = random.choice([
        '{} {}'.format(status, random.choice(beard_words)),
        '{} {} {}'.format(status, random.choice(['a', 'the']), random.choice(beard_words)),
        '{} {} face {} {}'.format(random.choice(change_actions),gender1,
            status, random.choice(beard_words)),
        '{} {} {} {}'.format(random.choice(change_actions),gender2,
            status, random.choice(beard_words))
    ])
    return txt


def diff2text(src_lab, tgt_lab, use_shuffle=True):
    gender1 = _get_gender(src_lab[gender_index])
    gender2 = random.choice(['him' if src_lab[gender_index] else 'her', 'it', 'everything'])
        
    if np.sum(np.abs(src_lab - tgt_lab)) == 0:
        txt = random.choice([
            'do nothing on {} face'.format(gender1),
            'do not {} anything'.format(random.choice(change_actions)),
            'keep {} unchanged'.format(gender2)
        ])
        return txt

    txt = []
    for idx, (src, trg) in enumerate(zip(src_lab, tgt_lab)):
        if idx in [0,1,2]:
            continue
        if idx == 3:
            txt.append(edit_gender(src, trg))
        if idx == 4:
            txt.append(edit_smiling(src, trg, tgt_lab[gender_index]))
        if idx == 5:
            txt.append(edit_age(src, trg, tgt_lab[gender_index]))
        if idx == 6:
            txt.append(edit_eyeglasses(src, trg, tgt_lab[gender_index]))
        if idx == 7:
            txt.append(edit_beard(src, trg, tgt_lab[gender_index]))

    txt.append(edit_hair_color(src_lab, tgt_lab))
    real_txt = []
    for it in txt:
        if it != '':
            real_txt.append(it)
    random.shuffle(real_txt)
    return ' . '.join(real_txt).strip()

def overall2text(tgt_lab, is_start=True):
    txt = ''
    a_or_an = random.choice(['a ', 'an '])

    if is_start:
        txt = random.choice(['', 'this is ', 'it is ']) 
    txt += a_or_an

    attr = [random.choice(smile_words) if tgt_lab[4] else random.choice(unsmile_words), 
            random.choice(young_words) if tgt_lab[5] else random.choice(old_words)]
    random.shuffle(attr)
    for at in attr:
        txt += at+' '

    txt += random.choice(male_words) if tgt_lab[3] else random.choice(female_words)
    txt += ' '
    target_color = np.nonzero(tgt_lab[:3])[0]
    color_txt = get_colors(target_color)
    status_glasses = random.choice(glasses_add) if tgt_lab[6] else random.choice(glasses_remove)
    status_beard = random.choice(beard_remove) if tgt_lab[7] else random.choice(beard_add)

    hair_txt = 'with {} hair'.format(color_txt)
    beard_txt = '{} {}'.format(status_beard, random.choice(beard_words))
    glasses_txt = '{} {}'.format(status_glasses, random.choice(glasses_words))
    hair_beard_glasses = [hair_txt, beard_txt, glasses_txt]
    random.shuffle(hair_beard_glasses)
    if random.random() > 0.5:
        txt += ' , '.join(hair_beard_glasses[:-1]) + ' and {}'.format(hair_beard_glasses[-1])
    else:
        txt += ' and '.join(hair_beard_glasses)
    return txt.strip()

def mixed2text(src_lab, tgt_lab):
    txt = '{} the '.format(random.choice(change_actions))
    attr = [random.choice(smile_words) if tgt_lab[4] else random.choice(unsmile_words), 
            random.choice(young_words) if tgt_lab[5] else random.choice(old_words)]
    random.shuffle(attr)
    for at in attr:
        if random.random() > 0.5:
            txt += at+' '

    txt += random.choice(male_words) if src_lab[3] else random.choice(female_words)
    txt += ' '
    source_color = np.nonzero(src_lab[:3])[0]
    src_color_txt = get_colors(source_color)
    status_glasses = random.choice(['with', 'wearing']) if src_lab[6] else 'without'
    status_beard = 'without' if src_lab[7] else random.choice(['with', 'wearing'])

    hair_txt = 'with {} hair'.format(src_color_txt)
    beard_txt = '{} {}'.format(status_beard, random.choice(beard_words))
    glasses_txt = '{} {}'.format(status_glasses, random.choice(glasses_words))
    hair_beard_glasses = [hair_txt, beard_txt, glasses_txt]
    random.shuffle(hair_beard_glasses)
    sub_att = []
    for at in hair_beard_glasses:
        if random.random() > 0.5:
            sub_att.append(at)
    if len(sub_att) > 0:
        if len(sub_att) < 3:
            txt += ' and '.join(sub_att)
        if len(sub_att) == 3:
            txt += ' , '.join(sub_att[:2]) + ' and {}'.format(sub_att[-1])
    txt += ' '
    txt += 'to '
    txt += overall2text(tgt_lab, False)
    return txt.strip()

def labels2text(src_lab, tgt_lab):
    return random.choice([
        diff2text(src_lab, tgt_lab),
        overall2text(tgt_lab),
        mixed2text(src_lab, tgt_lab)
    ]) + random.choice([" .", "", "?", " ", "!"])


if __name__ == '__main__':
    #'black', 'blond', 'brown', 'male', 'smile', 'young', 'eyeglasses', 'beard'
    src_labels = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    tgt_labels = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    print(diff2text(src_labels, tgt_labels))
    print(overall2text(tgt_labels))
    print(mixed2text(src_labels, tgt_labels))
    #print(automatic_text(src_labels, tgt_labels))


