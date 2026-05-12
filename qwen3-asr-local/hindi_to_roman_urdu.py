#!/usr/bin/env python3
"""
Devanagari (Hindi) → Roman Urdu transliterator
Direct single-step conversion — no intermediate Nastaliq.

Three-layer pipeline:
  Layer 1 — phoneme map (consonants, vowels, nuqta)
  Layer 2 — schwa deletion (word-final, virama-explicit, before non-Devanagari)
  Layer 3 — word endings normalisation + corrections dict

Usage:
    from hindi_to_roman_urdu import transliterate
    print(transliterate("मेरा नाम اکیب ہے"))   # "mera naam Aqib hai"

    python3 hindi_to_roman_urdu.py "मेरा नाम اکیب ہے"
"""

import re
import unicodedata

# ── Unicode constants ────────────────────────────────────────────────────────
NUKTA        = '़'  # ़  combining nukta
VIRAMA       = '्'  # ्  suppresses inherent vowel
ANUSVARA     = 'ं'  # ं  nasalisation
CHANDRABINDU = 'ँ'  # ँ  nasalisation
VISARGA      = 'ः'  # ः  (dropped in casual Urdu)

# Devanagari Unicode block boundaries — used for non-Devanagari boundary check
_DEVA_START = 0x0900
_DEVA_END   = 0x097F

# ── Layer 1: consonant tables ────────────────────────────────────────────────
CONSONANTS = {
    'क': 'k',   'ख': 'kh',  'ग': 'g',   'घ': 'gh',  'ङ': 'n',
    'च': 'ch',  'छ': 'ch',  'ज': 'j',   'झ': 'jh',  'ञ': 'n',
    'ट': 't',   'ठ': 'th',  'ड': 'd',   'ढ': 'dh',  'ण': 'n',
    'त': 't',   'थ': 'th',  'द': 'd',   'ध': 'dh',  'न': 'n',
    'प': 'p',   'फ': 'ph',  'ब': 'b',   'भ': 'bh',  'म': 'm',
    'य': 'y',   'र': 'r',   'ल': 'l',   'व': 'w',
    'श': 'sh',  'ष': 'sh',  'स': 's',   'ह': 'h',
    'ळ': 'l',
}

# base + nukta (U+093C) combinations — keyed as two-char strings
_N = NUKTA
NUKTA_MAP = {
    'क' + _N: 'q',   # क़  ka + nukta = qa
    'ख' + _N: 'kh',  # ख़  kha + nukta = kha (ghain-adjacent)
    'ग' + _N: 'gh',  # ग़  ga + nukta = ghain
    'ज' + _N: 'z',   # ज़  ja + nukta = za
    'ड' + _N: 'r',   # ड़  da + nukta = retroflex flap
    'ढ' + _N: 'rh',  # ढ़  dha + nukta
    'फ' + _N: 'f',   # फ़  pha + nukta = fa
    'य' + _N: 'y',   # य़  ya + nukta
}

# pre-composed extended range U+0958–U+095F (single codepoints)
EXTENDED = {
    'क़': 'q',   # क़
    'ख़': 'kh',  # ख़
    'ग़': 'gh',  # ग़
    'ज़': 'z',   # ज़
    'ड़': 'r',   # ड़
    'ढ़': 'rh',  # ढ़
    'फ़': 'f',   # फ़
    'य़': 'y',   # य़
}

# ── Layer 1: vowel tables ────────────────────────────────────────────────────
INDEPENDENT_VOWELS = {
    'अ': 'a',   'आ': 'aa',  'इ': 'i',   'ई': 'ee',
    'उ': 'u',   'ऊ': 'oo',  'ए': 'e',   'ऐ': 'ai',
    'ओ': 'o',   'औ': 'au',  'ऋ': 'ri',  'ॠ': 'ri',
    'ऍ': 'e',   'ऑ': 'o',   'ऌ': 'li',
}

MATRAS = {
    'ा': 'aa',  # ा
    'ि': 'i',   # ि
    'ी': 'ee',  # ी
    'ु': 'u',   # ु
    'ू': 'oo',  # ू
    'ृ': 'ri',  # ृ
    'े': 'e',   # े
    'ै': 'ai',  # ै
    'ो': 'o',   # ो
    'ौ': 'au',  # ौ
    'ॅ': 'e',   # ॅ
    'ॉ': 'o',   # ॉ
}

# word-boundary punctuation — schwa deleted before these
_PUNCT_BOUNDARY = set(' \t\n.,!?;:।॥"\'()[]{}')

# ── Layer 3: corrections dict ────────────────────────────────────────────────
# keys = phonetic output after layers 1+2+endings normalisation
# values = natural Roman Urdu spelling
CORRECTIONS = {
    # grammar particles
    # See ard/roman-urdu-convention.md for the full convention this dict enforces.

    # ── Pronouns ─────────────────────────────────────────────────────────
    'ham':         'hum',
    'wah':         'woh',
    'wo':          'woh',
    'yah':         'yeh',
    'ye':          'yeh',
    'too':         'tu',
    'mae':         'main',
    'me':          'main',
    'meraa':       'mera',
    'teraa':       'tera',
    'hamaaraa':    'hamara',
    'hamaara':     'hamara',
    'tumhaaraa':   'tumhara',
    'tumhaara':    'tumhara',
    'aapakaa':     'aapka',
    'aapaki':      'aapki',
    'unakaa':      'unka',
    'unaki':       'unki',
    'isakaa':      'iska',
    'usakaa':      'uska',
    'apanaa':      'apna',
    'apana':       'apna',
    'apanee':      'apni',
    'apani':       'apni',

    # ── Question words ───────────────────────────────────────────────────
    'kyaa':        'kya',
    'kaun':        'kaun',
    'kahaan':      'kahan',
    'wahaan':      'wahan',
    'yahaan':      'yahan',
    'kab':         'kab',
    'kaise':       'kaise',
    'kyon':        'kyon',
    'kyonki':      'kyonki',
    'kitanaa':     'kitna',
    'kitnaa':      'kitna',
    'kitanee':     'kitni',
    'kitnee':      'kitni',
    'kahin':       'kahin',

    # ── Conjunctions / connectors ────────────────────────────────────────
    'aur':         'aur',
    'lekin':       'lekin',
    'magar':       'magar',
    'agar':        'agar',
    'warna':       'warna',
    'phir':        'phir',
    'isalie':      'isliye',
    'isliye':      'isliye',
    'kelye':       'keliye',
    'kelie':       'keliye',
    'jab':         'jab',
    'tab':         'tab',
    'taki':        'taki',
    'taa':         'ta',

    # ── Postpositions ────────────────────────────────────────────────────
    'men':         'mein',
    'meen':        'mein',
    'tak':         'tak',
    'pae':         'pe',

    # ── Negation / affirmation ───────────────────────────────────────────
    'nahin':       'nahi',

    # Urdu wait/patience words (Hindi उ→i common in Urdu speech)
    'intazaar':    'intezar',
    'intazar':     'intezar',
    'intzaar':     'intezar',
    'sabra':       'sabar',
    'jaldee':      'jaldi',
    'foran':       'foran',
    'aakhirakaar': 'aakhirkar',
    'nahiin':      'nahi',
    'mat':         'mat',
    'haan':        'haan',
    'ji':          'ji',

    # ── Numbers ──────────────────────────────────────────────────────────
    'chhah':       'chhe',
    'chhe':        'chhe',
    'paanch':      'panch',
    'chaar':       'char',
    'gyaarah':     'gyarah',
    'baarah':      'barah',
    'tairah':      'terah',
    'chaudaa':     'chaudah',
    'pandrah':     'pandrah',
    'solah':       'solah',
    'satarah':     'satrah',
    'atharah':     'atharah',
    'unnees':      'unnees',
    'bees':        'bees',
    'pachas':      'pachas',
    'sau':         'sau',
    'hazaar':      'hazar',
    'laakh':       'lakh',
    'karor':       'crore',

    # ── Common verbs (-ना infinitives mostly handled by schwa rule;
    #     these are the irregular ones) ───────────────────────────────────
    'peena':       'pina',
    'jeena':       'jina',
    'seekhna':     'seekhna',
    'siikhana':    'seekhna',
    'paaonga':     'paonga',
    'denaa':       'dena',
    'denee':       'deni',
    'lenaa':       'lena',
    'lenee':       'leni',
    'aanaa':       'aana',
    'ulati':       'ulti',

    # ── Verb tense fragments / vowel hiatus (aa+e, ee+o, etc.) ───────────
    'jaaoge':      'jaoge',
    'aaoge':       'aoge',
    'jaaoonga':    'jaonga',
    'aaoonga':     'aonga',
    'kaaoge':      'kaoge',
    'aae':         'aaye',
    'aaen':        'ayen',
    'aaenge':      'ayenge',
    'gaae':        'gaye',
    'jaae':        'jaaye',
    'jaaen':       'jayen',
    'jaaenge':     'jayenge',
    'kaaenge':     'kayenge',
    'peeoge':      'peoge',
    'peeo':        'peo',
    'jeeoge':      'jeoge',
    'leeoge':      'loge',
    'deeoge':      'doge',
    'huaa':        'hua',
    'huee':        'hui',
    'huve':        'hue',

    # ── Family ───────────────────────────────────────────────────────────
    'maan':        'maa',
    'baap':        'baap',
    'ammee':       'ammi',
    'abbu':        'abbu',
    'beti':        'beti',
    'beta':        'beta',
    'bhaaee':      'bhai',
    'bhaai':       'bhai',
    'behan':       'behan',
    'bahan':       'behan',
    'daadaa':      'dada',
    'daadi':       'dadi',
    'naanaa':      'nana',
    'naani':       'nani',
    'chachaa':     'chacha',
    'chachi':      'chachi',
    'maamaa':      'mama',
    'maami':       'mami',
    'mausi':       'mausi',
    'phuphi':      'phuphi',

    # ── Body parts ───────────────────────────────────────────────────────
    'sir':         'sir',
    'aankh':       'aankh',
    'naak':        'naak',
    'kaan':        'kaan',
    'munh':        'munh',
    'daant':       'daant',
    'haath':       'haath',
    'paer':        'paer',
    'paaer':       'paer',
    'ungalee':     'ungli',
    'dil':         'dil',
    'peet':        'peet',

    # ── Food / drink ─────────────────────────────────────────────────────
    'roti':        'roti',
    'daal':        'daal',
    'chaawal':     'chawal',
    'sabzee':      'sabzi',
    'gosht':       'gosht',
    'machhalee':   'machhli',
    'andaa':       'anda',
    'doodh':       'doodh',
    'pani':        'pani',
    'chaae':       'chai',
    'chaay':       'chai',
    'qahwaa':      'qahwa',
    'mithaaee':    'mithai',
    'mithaai':     'mithai',
    'meethaai':    'mithai',
    'halwaa':      'halwa',
    'kheer':       'kheer',
    'biryaani':    'biryani',
    'pulaao':      'pulao',

    # ── Household ────────────────────────────────────────────────────────
    'ghar':        'ghar',
    'kamaraa':     'kamra',
    'darawaazaa':  'darwaza',
    'darwaazaa':   'darwaza',
    'khirakee':    'khirki',
    'chhat':       'chhat',
    'deewaar':     'deewar',
    'kursee':      'kursi',
    'mez':         'mez',
    'palang':      'palang',

    # ── Time ─────────────────────────────────────────────────────────────
    'waqat':       'waqt',
    'din':         'din',
    'raat':        'raat',
    'subah':       'subah',
    'subaah':      'subah',
    'dopahar':     'dopahar',
    'shaam':       'shaam',
    'mahinaa':     'mahina',
    'saal':        'saal',
    'aaj':         'aaj',
    'kal':         'kal',
    'paraso':      'parso',
    'parason':     'parson',
    'abhee':       'abhi',
    'kabhee':      'kabhi',
    'haameshaa':   'hamesha',
    'hameshaa':    'hamesha',
    'kabhi':       'kabhi',

    # ── Nature / world ───────────────────────────────────────────────────
    'aasamaan':    'aasman',
    'aasmaan':     'aasman',
    'dharatee':    'dharti',
    'sooraj':      'sooraj',
    'chaand':      'chand',
    'sitaaraa':    'sitara',
    'hawaa':       'hawa',
    'baadal':      'baadal',
    'baarish':     'baarish',
    'samandar':    'samandar',
    'pahaar':      'pahar',

    # ── Adjectives — qualities ───────────────────────────────────────────
    'achhaa':      'acha',
    'achchhaa':    'acha',
    'achcha':      'acha',
    'achch':       'ach',
    'bachcha':     'bacha',
    'bachchi':     'bachi',
    'pachcha':     'pacha',
    'pachchha':    'pacha',
    'kachcha':     'kacha',
    'sachcha':     'sacha',
    'buraa':       'bura',
    'sundar':      'sundar',
    'khoob':       'khoob',
    'kharaab':     'kharab',
    'aasaan':      'aasan',
    'mushkil':     'mushkil',
    'nayaa':       'naya',
    'puraanaa':    'purana',
    'garm':        'garam',
    'thandaa':     'thanda',
    'garmee':      'garmi',
    'sardee':      'sardi',
    'pyaar':       'pyar',
    'pyaaraa':     'pyara',
    'taazaa':      'taza',
    'meethaa':     'mitha',
    'meetha':      'mitha',
    'kadawaa':     'karwa',
    'tikhaa':      'tikha',

    # ── Adjectives — sizes ───────────────────────────────────────────────
    'baraa':       'bara',
    'baree':       'bari',
    'chhotaa':     'chota',
    'chhotee':     'choti',
    'lambaa':      'lamba',
    'lambee':      'lambi',
    'motaa':       'mota',
    'patalaa':     'patla',
    'gehraa':      'gehra',

    # ── Adjectives — colors ──────────────────────────────────────────────
    'kaalaa':      'kala',
    'safed':       'safed',
    'laal':        'laal',
    'peelaa':      'peela',
    'neelaa':      'neela',
    'haraa':       'hara',
    'naarangee':   'narangi',
    'jaamuni':     'jamuni',

    # ── Adverbs / quantifiers ────────────────────────────────────────────
    'bahut':       'bahut',
    'thoraa':      'thora',
    'thoree':      'thori',
    'zyaadaa':     'zyada',
    'zyaada':      'zyada',
    'kam':         'kam',
    'saaraa':      'sara',
    'kuchh':       'kuch',
    'koi':         'koi',
    'kuchhh':      'kuch',
    'sab':         'sab',
    'har':         'har',
    'sirph':       'sirf',
    'jaldee':      'jaldi',
    'dheere':      'dheere',
    'achaanak':    'achanak',

    # ── Religious / cultural ─────────────────────────────────────────────
    'salaam':      'salam',
    'duaa':        'dua',
    'namaaz':      'namaz',
    'masjid':      'masjid',
    'mandir':      'mandir',
    'khudaa':      'khuda',
    'allaah':      'Allah',
    'bhagawaan':   'bhagwan',
    'eed':         'Eid',
    'ramazaan':    'Ramzan',
    'shukariyaa':  'shukriya',
    'mehrabaani':  'meharbani',
    'mehrabaanee': 'meharbani',
    'insaaaan':    'insaan',
    'inshaalla':   'inshaAllah',
    'maashalla':   'mashaAllah',

    # ── ज्ञ conjunct fallback ─────────────────────────────────────────────
    'gyaan':       'gyan',
    'gyaani':      'gyani',
    'wigyaan':     'vigyan',
    'jnaan':       'gyan',

    # ── ड़ cluster overcorrection ─────────────────────────────────────────
    'laraki':      'larki',
    'larakee':     'larki',
    'laraka':      'larka',
    'larakaa':     'larka',

    # ── ि + ए glide → 'iye' not 'ie' ─────────────────────────────────────
    'die':         'diye',
    'lie':         'liye',
    'kie':         'kiye',
    'pie':         'piye',
    'chaahie':     'chahiye',     # चाहिए — need / want
    'chaahiye':    'chahiye',
    'aaie':        'aaiye',       # आइए — please come
    'baithie':     'baithiye',    # बैठिए — please sit
    'jaaie':       'jaaiye',      # जाइए — please go
    'kahie':       'kahiye',      # कहिए — please say
    'sunie':       'suniye',      # सुनिए — please listen
    'dekhie':      'dekhiye',     # देखिए — please look

    # -ीजिए polite imperative pattern: keejie→kijiye, dijie→dijiye, etc.
    'keejie':      'kijiye',      # कीजिए — please do
    'dijie':       'dijiye',      # दीजिए — please give
    'lijie':       'lijiye',      # लीजिए — please take
    'pijie':       'pijiye',      # पीजिए — please drink
    'sunijie':     'suniye',      # सुनिए (variants)
    'kahijie':     'kahiye',      # कहिए (variants)
    'dekhijie':    'dekhiye',     # देखिए (variants)
    'jaaijie':     'jaaiye',      # जाइए (variants)

    # ── फ in Urdu loan words → 'f' not 'ph' ──────────────────────────────
    'tarph':       'taraf',
    'pharq':       'farq',
    'pharaq':      'farq',
    'pharz':       'farz',
    'pharaz':      'farz',
    'phaur':       'faur',
    'phauj':       'fauj',
    'phaisla':     'faisla',
    'philam':      'film',
    'phon':        'fon',
    'phaisalaa':   'faisla',

    # ── word-initial 'aa' shortening in common words ─────────────────────
    'aadaab':      'adaab',
    'aazaad':      'azaad',
    'aakhir':      'akhir',
    'aagaaz':      'aagaz',
    'aasaaan':     'aasan',
    'aaeeen':      'aaeen',

    # ── ASR-prone Urdu vs phonetic Hindi spellings ───────────────────────
    'zabaan':      'zaban',
    'zaabaan':     'zaban',
    'aawaaz':      'awaz',
    'aawaz':       'awaz',
    'aawaaj':      'awaz',
    'shinaakht':   'shanakht',
    'shinaakhat':  'shanakht',
    'mausaam':     'mausam',
    'zindagii':    'zindagi',
    'khushii':     'khushi',
    'duniyaa':     'duniya',
    'tabiyat':     'tabiyat',
    'tabeeyat':    'tabiyat',
    'aramaan':     'arman',
    'imaan':       'imaan',
    'shahar':      'shehr',
    'mulk':        'mulk',
    'jangal':      'jangal',
    'rasta':       'rasta',
    'raastaa':     'rasta',
    'rasataa':     'rasta',

    # ── greetings / phrases ──────────────────────────────────────────────
    'aadaab':      'adaab',
    'salaam':      'salam',
    'namaste':     'namaste',
    'khudaa hafiz':'khuda hafiz',
    'alavidaa':    'alvida',
    'theek':       'theek',
    'teek':        'theek',
    'maaph':       'maaf',
    'maaf':        'maaf',

    # ── English loan words — call center / customer service ─────────────
    # ASR transcribes English phonetically in Devanagari, transliterator
    # converts back to phonetic Roman; these map back to original English.
    'hello':       'hello',
    'helo':        'hello',
    'haelo':       'hello',
    'haay':        'hi',
    'hai':         'hai',           # native 'hai' (है) wins; English 'hi' is rarely written as 'hai' in Hindi
    'pleez':       'please',
    'plij':        'please',
    'sori':        'sorry',
    'sory':        'sorry',
    'okae':        'okay',
    'oke':         'okay',
    'ok':          'ok',
    'thank you':   'thank you',
    'thaink yoo':  'thank you',
    'thaink':      'thank',
    'wailkam':     'welcome',
    'yes':         'yes',
    'no':          'no',

    # ── Roles / titles ──────────────────────────────────────────────────
    'doktar':      'doctor',
    'daakatar':    'doctor',
    'naras':       'nurse',
    'stafa':       'staff',
    'staaf':       'staff',
    'maenejar':    'manager',
    'eajent':      'agent',
    'kastamar':    'customer',
    'sar':         'sir',           # 'sir' usually retained as English
    'maaeem':      'madam',
    'madam':       'madam',
    'saahib':      'sahib',         # साहिब — conventional shorter form
    'saahab':      'sahab',
    'saaheb':      'sahib',
    'janaab':      'janab',

    # ── Appointment / scheduling ────────────────────────────────────────
    'apointament': 'appointment',
    'apoinatement':'appointment',
    'apoyentament':'appointment',
    'buking':      'booking',
    'bookking':    'booking',
    'kainsal':     'cancel',
    'kainsil':     'cancel',
    'kainsel':     'cancel',
    'kanfarm':     'confirm',
    'kanphirm':    'confirm',
    'shedyool':    'schedule',
    'shedyul':     'schedule',
    'reshedyool':  'reschedule',
    'rishedyul':   'reschedule',
    'pik ap':      'pick up',
    'pikaap':      'pick up',
    'pik':         'pick',
    'ap':          'up',           # 'अप' English prefix; conflicts rare
    'kalekt':      'collect',
    'kolekt':      'collect',
    'drap':        'drop',
    'dileewari':   'delivery',
    'deliwari':    'delivery',

    # ── Lab / medical procedures ────────────────────────────────────────
    'blad':        'blood',
    'blud':        'blood',
    'sainpal':     'sample',
    'sampal':      'sample',
    'sample':      'sample',
    'riport':      'report',
    'reepart':     'report',
    'rizalt':      'result',
    'reesalt':     'result',
    'laib':        'lab',
    'leb':         'lab',
    'kalekshan':   'collection',
    'colection':   'collection',
    'chekaap':     'checkup',
    'cheekap':     'checkup',
    'skain':       'scan',
    'sken':        'scan',
    'eksaray':     'x-ray',
    'ksray':       'x-ray',
    'eemaaraaee':  'MRI',
    'seetee skain':'CT scan',
    'eesheejee':   'ECG',
    'aalterasaaund':'ultrasound',
    'alterasaund': 'ultrasound',
    'baiopasi':    'biopsy',

    # ── Tests / lab parameters ──────────────────────────────────────────
    'shugar':      'sugar',
    'kolestrol':   'cholesterol',
    'koleestrol':  'cholesterol',
    'thaayaroid':  'thyroid',
    'thayaroid':   'thyroid',
    'witaamin':    'vitamin',
    'vitamin':     'vitamin',
    'faasting':    'fasting',
    'phaasting':   'fasting',
    'yoorin':      'urine',
    'urin':        'urine',
    'haymoglobin': 'hemoglobin',
    'hemoglobin':  'hemoglobin',
    'eemyoon':     'immune',
    'liwar':       'liver',
    'kidanee':     'kidney',
    'haart':       'heart',
    'lang':        'lung',

    # ── Contact info / logistics ────────────────────────────────────────
    'fon':         'phone',
    'phon':        'phone',
    'nanbar':      'number',
    'nambar':      'number',
    'edres':       'address',
    'aidres':      'address',
    'imaaeel':     'email',
    'eemel':       'email',
    'iemel':       'email',
    'taaim':       'time',
    'time':        'time',
    'det':         'date',
    'deet':        'date',
    'eria':        'area',
    'area':        'area',
    'rod':         'road',
    'satriit':     'street',
    'striit':      'street',
    'haaus':       'house',
    'eparteement': 'apartment',
    'flait':       'flat',

    # ── Payment / money ─────────────────────────────────────────────────
    'paese':       'paise',         # native paise stays
    'rupye':       'rupaye',        # रुपये — both used, 'rupaye' is fuller
    'rupayaa':     'rupaya',
    'paisaa':      'paisa',
    'kaiash':      'cash',
    'kaesh':       'cash',
    'kraidit kaard':'credit card',
    'kard':        'card',
    'aanlaain':    'online',
    'pemement':    'payment',

    # ── Common conversational English loans ─────────────────────────────
    'taim':        'time',
    'haam':        'home',
    'hom':         'home',
    'aafis':       'office',
    'ophis':       'office',
    'shap':        'shop',
    'maarket':     'market',
    'mobaaeel':    'mobile',
    'mobile':      'mobile',
    'kaal':        'call',
    'masej':       'message',
    'mesaej':      'message',
    'wats ap':     'WhatsApp',
    'whatsaap':    'WhatsApp',
}


# ── Proper nouns dict (names — preserve / add capitalisation) ───────────────
# ASR transcribes names phonetically in Devanagari, so we need explicit mapping
# back to conventional Roman Urdu/English spellings.
PROPER_NOUNS = {
    # ── Common Muslim/Urdu male names ────────────────────────────────────
    # عاقب — the user's name; ASR mishears as 'akeeb', 'aqeeb', 'aakib' etc.
    'akeeb':       'Aqib',
    'aqeeb':       'Aqib',
    'akib':        'Aqib',
    'aqib':        'Aqib',
    'aaqib':       'Aqib',
    'aakib':       'Aqib',
    # علی
    'ali':         'Ali',
    'alee':        'Ali',
    # عمر
    'umar':        'Umar',
    'omar':        'Umar',
    # عثمان
    'usman':       'Usman',
    'usmaan':      'Usman',
    'uthman':      'Usman',
    # محمد
    'muhammad':    'Muhammad',
    'mohammad':    'Muhammad',
    'mohammed':    'Muhammad',
    'muhammed':    'Muhammad',
    # احمد
    'ahmad':       'Ahmad',
    'ahamad':      'Ahmad',
    'ahmed':       'Ahmad',
    'ahamed':      'Ahmad',
    # حسن
    'hassan':      'Hassan',
    'hasan':       'Hassan',
    # حسین
    'hussain':     'Hussain',
    'husain':      'Hussain',
    'husein':      'Hussain',
    # نعیم
    'naeem':       'Naeem',
    'naieem':      'Naeem',
    # یوسف
    'yousaf':      'Yousaf',
    'yusuf':       'Yusuf',
    # ابراہیم
    'ibrahim':     'Ibrahim',
    'ibraheem':    'Ibrahim',
    # اسماعیل
    'ismail':      'Ismail',
    'ismaeel':     'Ismail',
    # طارق
    'tariq':       'Tariq',
    'taariq':      'Tariq',
    # عمران
    'imran':       'Imran',
    'imraan':      'Imran',
    # کامران
    'kamran':      'Kamran',
    'kaamran':     'Kamran',
    # سلمان
    'salman':      'Salman',
    'salmaan':     'Salman',
    # عارف
    'arif':        'Arif',
    'aarif':       'Arif',
    # آصف
    'asif':        'Asif',
    'aasif':       'Asif',
    # کاشف
    'kashif':      'Kashif',
    'kaashif':     'Kashif',
    # شاہد
    'shahid':      'Shahid',
    'shaahid':     'Shahid',
    # راشد
    'rashid':      'Rashid',
    'raashid':     'Rashid',
    # خالد
    'khalid':      'Khalid',
    'khaalid':     'Khalid',
    # مجید
    'majeed':      'Majeed',
    'majid':       'Majeed',
    # بلال
    'bilal':       'Bilal',
    'bilaal':      'Bilal',
    # عبداللہ
    'abdullah':    'Abdullah',
    'abdulla':     'Abdullah',
    # ندیم
    'nadeem':      'Nadeem',
    # وسیم
    'waseem':      'Waseem',
    'wasim':       'Waseem',
    # فیصل
    'faisal':      'Faisal',
    'phaisal':     'Faisal',

    # ── Common Muslim/Urdu female names ──────────────────────────────────
    # عائشہ
    'ayesha':      'Ayesha',
    'aisha':       'Ayesha',
    'aaisha':      'Ayesha',
    # فاطمہ
    'fatima':      'Fatima',
    'faatima':     'Fatima',
    # مریم
    'maryam':      'Maryam',
    'mariyam':     'Maryam',
    # خدیجہ
    'khadija':     'Khadija',
    'khadijah':    'Khadija',
    # زینب
    'zainab':      'Zainab',
    'zaynab':      'Zainab',
    # سعدیہ
    'sadia':       'Sadia',
    'saadia':      'Sadia',
    'sadiya':      'Sadia',
    # آمنہ
    'amna':        'Amna',
    'amnaa':       'Amna',
    'aamna':       'Amna',
    # سارہ — name removed from auto-mapping: conflicts with adjective सारा ("all/whole")
    # If you need the female name, capitalise manually after transliteration.
    # صفیہ
    'safia':       'Safia',
    'safiya':      'Safia',
    # رابعہ
    'rabia':       'Rabia',
    'raabia':      'Rabia',

    # Place names
    'karaachee':   'Karachi',
    'karachee':    'Karachi',
    'karaanchi':   'Karachi',
    'karaachi':    'Karachi',
    'lahaur':      'Lahore',
    'islaamaabaad':'Islamabad',
    'islamabad':   'Islamabad',
    'paakistaan':  'Pakistan',
    'hindustaan':  'Hindustan',
    'dilli':       'Delhi',
    'dillee':      'Delhi',
    'mumbaee':     'Mumbai',
    'mumbai':      'Mumbai',
}


# ── Core algorithm ────────────────────────────────────────────────────────────

def _is_deva(ch: str) -> bool:
    return _DEVA_START <= ord(ch) <= _DEVA_END


def _emit_vowel(chars: list, i: int, n: int, roman_c: str, result: list) -> int:
    """
    After consuming a consonant, determine and emit roman_c + its vowel.
    Schwa (inherent 'a') is deleted when:
      - end of string
      - followed by virama (explicit suppression)
      - followed by punctuation / space
      - followed by a non-Devanagari character (digit, ASCII, Arabic script, etc.)
    Returns updated index.
    """
    if i >= n:
        result.append(roman_c)
        return i

    ch = chars[i]

    # explicit virama — no vowel
    if ch == VIRAMA:
        result.append(roman_c)
        return i + 1

    # dependent vowel matra
    if ch in MATRAS:
        vowel = MATRAS[ch]
        i += 1
        if i < n and chars[i] in (ANUSVARA, CHANDRABINDU):
            # ī + nasalisation → 'in' (नहीं→nahin, कहीं→kahin, यहीं→yahin)
            # other long vowels keep their length (हाँ→haan, हूँ→hoon, हैं→hain)
            if vowel == 'ee':
                result.append(roman_c + 'in')
            else:
                result.append(roman_c + vowel + 'n')
            return i + 1
        result.append(roman_c + vowel)
        return i

    # anusvara / chandrabindu directly after consonant → inherent 'a' + nasal
    if ch in (ANUSVARA, CHANDRABINDU):
        result.append(roman_c + 'an')
        return i + 1

    # punctuation or space — word boundary, delete schwa
    if ch in _PUNCT_BOUNDARY:
        result.append(roman_c)
        return i

    # non-Devanagari character (digit, ASCII, Urdu Nastaliq, etc.) — word boundary
    if not _is_deva(ch):
        result.append(roman_c)
        return i

    # ── Schwa syncope before a final consonant+matra cluster ─────────────
    # Pattern: C + C + matra + (boundary)  →  delete C's inherent schwa
    # Fires on verb infinitives and similar: करना→karna, आदमी→aadmi, बोलना→bolna
    # Guard: only fires when a vowel has already been emitted in this word.
    # Otherwise it would over-delete first-syllable schwas (नदी→ndi, बड़ा→bra).
    # Does NOT fire when:
    #   - we're at the first consonant of the word (no vowel before)
    #   - next consonant has no matra (जगह→jagah stays with schwas)
    #   - matra is not word-final (मनाना: first matra has another consonant after)
    if (ch in CONSONANTS or ch in EXTENDED) \
            and result and result[-1] and result[-1][-1] in 'aeiouy':
        j = i + 1
        if j < n and chars[j] == NUKTA:
            j += 1
        if j < n and chars[j] in MATRAS:
            k = j + 1
            if k >= n or chars[k] in _PUNCT_BOUNDARY or not _is_deva(chars[k]):
                result.append(roman_c)
                return i

    # medial position within Devanagari word — keep inherent 'a'
    result.append(roman_c + 'a')
    return i


def _transliterate_raw(text: str) -> str:
    """Layer 1 + 2: character-level phoneme mapping with schwa deletion."""
    text = unicodedata.normalize('NFC', text)
    chars = list(text)
    result = []
    i = 0
    n = len(chars)

    while i < n:
        ch = chars[i]

        # ── Devanagari digits → ASCII ────────────────────────────────────
        if '०' <= ch <= '९':
            result.append(str(ord(ch) - 0x0966))
            i += 1
            continue

        # ── Pre-composed extended nuqta consonants (U+0958–U+095F) ───────
        if ch in EXTENDED:
            roman_c = EXTENDED[ch]
            i += 1
            i = _emit_vowel(chars, i, n, roman_c, result)
            continue

        # ── Special conjunct: ज् + ञ → 'gy' ─────────────────────────────
        if (ch == 'ज' and i + 2 < n
                and chars[i + 1] == VIRAMA and chars[i + 2] == 'ञ'):
            i += 3
            i = _emit_vowel(chars, i, n, 'gy', result)
            continue

        # ── Regular consonants ────────────────────────────────────────────
        if ch in CONSONANTS:
            roman_c = CONSONANTS[ch]
            # combining nukta follows → override mapping
            if i + 1 < n and chars[i + 1] == NUKTA:
                key = ch + NUKTA
                roman_c = NUKTA_MAP.get(key, roman_c)
                i += 1  # consume nukta
            i += 1
            i = _emit_vowel(chars, i, n, roman_c, result)
            continue

        # ── Independent vowels ────────────────────────────────────────────
        if ch in INDEPENDENT_VOWELS:
            result.append(INDEPENDENT_VOWELS[ch])
            i += 1
            if i < n and chars[i] in (ANUSVARA, CHANDRABINDU):
                result.append('n')
                i += 1
            continue

        # ── Standalone anusvara / chandrabindu ───────────────────────────
        if ch in (ANUSVARA, CHANDRABINDU):
            result.append('n')
            i += 1
            continue

        # ── Visarga — drop ────────────────────────────────────────────────
        if ch == VISARGA:
            i += 1
            continue

        # ── Devanagari danda ─────────────────────────────────────────────
        if ch in ('।', '॥'):  # । ॥
            result.append('.')
            i += 1
            continue

        # ── Pass through (ASCII, spaces, Urdu Nastaliq, etc.) ────────────
        result.append(ch)
        i += 1

    return ''.join(result)


def _normalize_endings(text: str) -> str:
    """
    Normalise long vowels to natural Roman Urdu conventions.

    Rules applied in order:
      1. word-final 'ee' → 'i'   (paanee→paani, nadee→nadi)
      2. word-final 'aa' → 'a'   (khaanaa→khaana, meraa→mera)
      3. aa + consonant + vowel at word-end → a + consonant + vowel
         Runs AFTER rules 1+2 so it sees 'paani'/'khaana' not 'paanee'/'khaanaa'
         Handles: paani→pani, khaana→khana, bataana→batana
         Does NOT affect: naam ('m' not followed by vowel), pyaar (corrected separately)
    """
    _C = '[bcdfghjklmnpqrstvwxyz]'
    _V = '[aeiouy]'
    text = re.sub(r'ee\b', 'i', text)
    # word-final 'aa' → 'a' ONLY when preceded by a consonant
    # (otherwise standalone आ would wrongly shorten: "कब आ" → "kab a")
    text = re.sub(rf'(?<={_C})aa\b', 'a', text)
    text = re.sub(rf'aa({_C}{_V})\b', r'a\1', text)
    return text


def _apply_corrections(text: str) -> str:
    """
    Layer 3: replace known wrong phonetic words with natural Roman Urdu.
    Two lookup tables:
      - PROPER_NOUNS: names / places — capitalisation taken from dict value
      - CORRECTIONS:  common words — preserves original word's capitalisation
    PROPER_NOUNS wins when the same key exists in both.
    """
    def fix_word(m):
        w = m.group(0)
        lower = w.lower()
        # Proper nouns: explicit capitalisation in the mapping
        if lower in PROPER_NOUNS:
            return PROPER_NOUNS[lower]
        # Common-word corrections: keep original capitalisation
        corrected = CORRECTIONS.get(lower)
        if not corrected:
            return w
        if w[0].isupper():
            return corrected[0].upper() + corrected[1:]
        return corrected

    return re.sub(r'[A-Za-z]+', fix_word, text)


def transliterate(text: str) -> str:
    """
    Convert Hindi Devanagari text to Roman Urdu.
    Non-Devanagari characters (ASCII, Urdu Nastaliq, digits) pass through unchanged.
    """
    raw    = _transliterate_raw(text)
    normed = _normalize_endings(raw)
    return _apply_corrections(normed)


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    tests = [
        # (input, expected)
        ("मेरा नाम اکیب ہے۔",            "mera naam اکیب ہے۔"),
        ("आज का मौसम بہت اچھا ہے۔",      "aaj ka mausam بہت اچھا ہے۔"),
        ("बहुत अच्छा है।",                "bahut acha hai."),
        ("यह कोड की ज़बान में आवाज़ की शिनाख़्त का टेस्ट है।",
                                           "yeh kod ki zaban mein awaz ki shanakht ka test hai."),
        # vowel ending normalisation
        ("नदी",      "nadi"),
        ("ज़िंदगी",  "zindagi"),
        ("बड़ा",     "bara"),
        ("छोटा",     "chota"),
        ("पानी",     "pani"),
        ("खाना",     "khana"),
        ("बताना",    "batana"),
        ("लड़की",    "larki"),
        ("प्यार",    "pyar"),
        ("आसमान",    "aasman"),
        # conjuncts / clusters
        ("ज्ञान",    "gyan"),
        ("धर्म",     "dharm"),
        ("शुक्रिया", "shukriya"),
        # function words
        ("हम",       "hum"),
        ("आवाज",     "awaz"),
        ("आवाज़",    "awaz"),
        ("ज़बान",    "zaban"),
        # stable words that must NOT change
        ("नाम",      "naam"),
        ("हूँ",      "hoon"),
        ("हैं",      "hain"),
        ("रात",      "raat"),
        ("काम",      "kaam"),
        # digits + mixed
        ("नाम123",   "naam123"),
        # schwa syncope (verb infinitives + CCV-matra word-end)
        ("करना",     "karna"),
        ("देखना",    "dekhna"),
        ("सुनना",    "sunna"),
        ("बोलना",    "bolna"),
        ("लिखना",    "likhna"),
        ("समझना",    "samajhna"),
        ("आदमी",     "aadmi"),
        # first-syllable schwa MUST NOT delete (regression guard)
        ("नदी",      "nadi"),
        ("बड़ा",     "bara"),
        ("गली",      "gali"),
        # consonant cluster cleanup
        ("बच्चा",    "bacha"),
        # corrections
        ("इसलिए",    "isliye"),
        ("सिर्फ",    "sirf"),
        ("आदाब",     "adaab"),
        ("क्या तुम जाओगे?", "kya tum jaoge?"),
        # CC word-final without matra (must keep schwa)
        ("जगह",      "jagah"),
        # ── Proper nouns (PROPER_NOUNS dict) ─────────────────────────────
        ("अकीब",     "Aqib"),
        ("अली",      "Ali"),
        ("मोहम्मद",  "Muhammad"),
        ("अहमद",     "Ahmad"),
        ("करांची",   "Karachi"),
        ("मेरा नाम अकीब है।", "mera naam Aqib hai."),
        # ── Comprehensive corrections (CORRECTIONS dict) ─────────────────
        ("चार",      "char"),
        ("पाँच",     "panch"),
        ("कितना",    "kitna"),
        ("कितनी",    "kitni"),
        ("बेटा",     "beta"),
        ("बेटी",     "beti"),
        ("भाई",      "bhai"),
        ("बहन",      "behan"),
        ("नया",      "naya"),
        ("ठंडा",     "thanda"),
        ("गरम",      "garam"),
        # ── Vowel hiatus (aa+e, ee+o) ────────────────────────────────────
        ("क्या तुम पानी पीओगे?", "kya tum pani peoge?"),
        ("हम कल जाएंगे।",        "hum kal jayenge."),
    ]

    if len(sys.argv) > 1:
        print(transliterate(' '.join(sys.argv[1:])))
    else:
        print("── Self-test ──────────────────────────────────")
        passed = 0
        for inp, expected in tests:
            result = transliterate(inp)
            ok = result == expected
            if ok:
                passed += 1
            mark = '✓' if ok else '✗'
            print(f"{mark}  {inp!r:35} → {result!r:30}  (expected {expected!r})")
        print(f"\n{passed}/{len(tests)} passed")
