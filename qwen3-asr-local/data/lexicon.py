"""
Hindi → Roman Urdu lexicon (spelling overrides)

Two maintenance dicts consumed by hindi_to_roman_urdu.py:

  CORRECTIONS  — common words AND multi-word phrases
                 keys without spaces → word-level lookup
                 keys with spaces    → phrase-level replacement
                 (raw phonetic output after layers 1+2+endings normalisation
                  → conventional Roman Urdu spelling)

  PROPER_NOUNS — names, places, acronyms, brands
                 explicit capitalisation taken from dict value
                 (raw phonetic → properly-capitalised name)

To add a new correction:
  1. Run the wrong sentence through transliterator and note the raw word
  2. Add 'raw_word': 'corrected_word' to CORRECTIONS
     (or PROPER_NOUNS if it's a name/place/acronym needing caps)

See ard/transliterator-findings.md for the full workflow.
"""

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
    'aaen':        'aayen',
    'aaenge':      'aayenge',
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
    'jaaie':       'jaiye',       # जाइए — please go (shorter Urdu convention)
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
    'jaaijie':     'jaiye',       # जाइए (variants)

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

    # ── More English loan words ──────────────────────────────────────────
    'buk':         'book',
    'kol':         'call',
    'maisej':      'message',
    'weediyo':     'video',
    'aip':         'app',
    'phaail':      'file',
    'phaarm':      'form',
    'kaard':       'card',
    'paasaward':   'password',
    'pasawarda':   'password',
    'saain':       'sign',
    'aproow':      'approve',
    'sabamit':     'submit',
    'apalod':      'upload',
    'daaunalod':   'download',
    'sheyar':      'share',
    'kopi':        'copy',
    'pest':        'paste',
    'phaiks':      'fax',
    'baaik':       'bike',
    'taiksi':      'taxi',
    'hotal':       'hotel',
    'restaraan':   'restaurant',
    'baink':       'bank',
    'kophi':       'coffee',
    'joos':        'juice',
    'injeeniyar':  'engineer',
    'teechar':     'teacher',
    'pulis':       'police',
    'draaiwar':    'driver',
    'kuk':         'cook',
    'kook':        'cook',

    # ── Food words (Roman Urdu conventional shortening) ──────────────────
    'machli':      'machhli',   # मछली — fish, conventionally written 'machhli'
    'pulaaw':      'pulao',
    'kabaab':      'kabab',
    'chatni':      'chutney',
    'achaar':      'achar',
    'birayani':    'biryani',

    # ── Body parts ───────────────────────────────────────────────────────
    'baazoo':      'baazu',
    'ungali':      'ungli',
    'pair':        'paer',
    'eri':         'edi',
    'jeebh':       'jeebh',

    # ── Emotions / feelings (long → short) ───────────────────────────────
    'pareshaan':   'pareshan',
    'napharat':    'nafrat',
    'hairaan':     'hairan',
    'phakhr':      'fakhr',
    'beemaar':     'bimar',
    'kamazor':     'kamzor',
    'majaboot':    'majboot',
    'akasar':      'aksar',

    # ── Time word shortening ─────────────────────────────────────────────
    'haphte':      'hafte',
    'maheene':     'mahine',
    'mahinaa':     'mahina',

    # ── Family (long ā→a, double consonant retained) ─────────────────────
    'abboo':       'abbu',
    'chaacha':     'chacha',
    'chaachi':     'chachi',
    'beewi':       'biwi',
    'daamaad':     'damad',

    # ── Professions (Urdu native + English loan) ─────────────────────────
    'phauji':      'fauji',
    'kisaan':      'kisan',
    'mazadoor':    'mazdoor',
    'chaparasi':   'chaprasi',
    'hajjaam':     'hajjam',
    'halawaai':    'halwai',
    'darazi':      'darzi',
    'wakeel':      'wakeel',

    # ── Colors ───────────────────────────────────────────────────────────
    'saphed':      'safed',
    'naarangi':    'narangi',
    'bhoora':      'bhura',
    'sunahra':     'sunehra',
    'baingani':    'baingani',

    # ── Adjectives (size, quality) ───────────────────────────────────────
    'lanba':       'lamba',     # लंबा — ं before ब prefers 'm' in Roman Urdu
    'gahra':       'gehra',     # गहरा — schwa→e in Urdu speech
    'phaaltoo':    'faltu',
    'mahanga':     'mehnga',    # महंगा — same schwa→e pattern
    'basi':        'baasi',     # बासी — internal aa kept
    'oonchaa':     'ooncha',
    'neechaa':     'neecha',
    'ameer':       'ameer',
    'gareeb':      'gareeb',

    # ── Verbs (schwa→e in Urdu speech; long ā→a in colloquial) ───────────
    'kahna':       'kehna',
    'jaanna':      'janna',     # जानना — long aa→a colloquial
    'maanna':      'manna',
    'daurna':      'daudna',
    'kaatna':      'katna',
    'maarna':      'marna',
    'haarna':      'harna',
    'gaana':       'gaana',     # already kept (singing/song)

    # ── Sentence-level patterns ──────────────────────────────────────────
    'jaaiye':      'jaiye',
    'taiyaar':     'tayyar',
    'taiyaaree':   'tayyari',
    'jaaegi':      'jayegi',
    'jaaega':      'jayega',
    'aaegi':       'aayegi',
    'aaega':       'aayega',
    # 'aaenge' and 'jaaenge' canonical entries are above (preserve long aa)
    'suwidha':     'suvidha',
    'wid':         'vid',
    'widya':       'vidya',
    'widyaalay':   'vidyalay',
    'wakt':        'waqt',
    'naheen':      'nahi',     # belt-and-suspenders fallback

    # ── Numbers (extended — internal aa shortening) ──────────────────────
    'athaarah':    'atharah',
    'ikkees':      'ikkis',
    'baaees':      'baees',
    'chaalees':    'chalees',
    'pachaas':     'pachas',
    'paintaalees': 'paintalees',
    'pachchees':   'pacchees',
    'paitees':     'paintees',
    'pahla':       'pehla',
    'paanchawaan': 'panchwan',
    'dhaai':       'dhai',
    'doosra':      'doosra',
    'teesra':      'teesra',
    'aadha':       'aadha',
    'pauna':       'pauna',
    'sawa':        'sawa',
    'derh':        'derh',

    # ── Medical conditions / symptoms ───────────────────────────────────
    'zukaam':      'zukam',
    'khaansi':     'khansi',
    # Body part + दर्द compounds — split with space (Roman Urdu convention)
    'saradard':    'sir dard',     # सरदर्द — head pain
    'siradard':    'sir dard',     # सिरदर्द (variant)
    'sardard':     'sir dard',     # already-corrected form maps too
    'petadard':    'pet dard',     # पेटदर्द — stomach pain
    'petdard':     'pet dard',
    'kamaradard':  'kamar dard',   # कमरदर्द — back pain
    'kamardard':   'kamar dard',
    'daantadard':  'daant dard',   # दाँतदर्द — tooth pain
    'daantdard':   'daant dard',
    'galedard':    'gale dard',    # गलेदर्द — throat pain
    'seenedard':   'seene dard',   # सीनेदर्द — chest pain
    'aankhadard':  'aankh dard',   # आँखदर्द — eye pain
    'kaanadard':   'kaan dard',    # कानदर्द — ear pain
    'kaandard':    'kaan dard',
    'kamazori':    'kamzori',
    'gais':        'gas',
    'esiditi':     'acidity',
    'preshar':     'pressure',
    'daayabiteez': 'diabetes',
    'haaiparatenshan': 'hypertension',
    'asthama':     'asthma',
    'hepataaitis': 'hepatitis',
    'elarji':      'allergy',
    'inphekshan':  'infection',
    'waayaral':    'viral',
    'baikteeriyal':'bacterial',
    'strok':       'stroke',
    'kainsar':     'cancer',
    'tyoomar':     'tumor',
    'dawaai':      'dawai',
    'injekshan':   'injection',
    'sirap':       'syrup',
    'daura':       'daura',

    # ── Lab tests (mostly acronyms — see PROPER_NOUNS for caps) ─────────
    'krietinin':   'creatinine',
    'yooriya':     'urea',
    'bileeroobin': 'bilirubin',
    'aayaran':     'iron',
    'kailshiyam':  'calcium',
    'maigneeshiyam':'magnesium',
    'sodiyam':     'sodium',
    'potaishiyam': 'potassium',
    'proteen':     'protein',
    'elbumin':     'albumin',
    'glookoz':     'glucose',
    'prolaiktin':  'prolactin',

    # ── Banking / money ─────────────────────────────────────────────────
    'khata':       'khaata',
    'akaaunt':     'account',
    'kaish':       'cash',
    'chek':        'check',
    'kredit':      'credit',
    'bailens':     'balance',
    'traansaphar': 'transfer',
    'withadro':    'withdraw',
    'dipozit':     'deposit',
    'lon':         'loan',
    'bil':         'bill',
    'pement':      'payment',
    'riphand':     'refund',
    'byaaj':       'byaaj',
    'keemat':      'keemat',
    'daam':        'daam',

    # ── Pronouns (extra forms) ──────────────────────────────────────────
    'hamen':       'hamein',
    'tumhen':      'tumhein',
    'inhen':       'inhein',
    'unhen':       'unhein',

    # ── Location / direction ────────────────────────────────────────────
    'peeche':      'peechhe',
    'daaen':       'daayen',
    'baaen':       'baayen',
    'aage':        'aage',
    'andar':       'andar',
    'baahar':      'baahar',

    # ── Verb forms (more) ───────────────────────────────────────────────
    'aonga':       'aaonga',
    'karoonga':    'karunga',
    'jata':        'jaata',       # जाता हूँ — verb form preserves long ā
    'ata':         'aata',        # आता — keep long ā (rarely needs override; safe)
    # causative future (-वाऊँगा/वाऊँगी): drop schwa between C-C, shorten oo→o
    'karawaaoonga':'karwaaonga',  # करवाऊँगा — will get done
    'karawaaoongi':'karwaaongi',
    'dikhaaoonga': 'dikhaaonga',
    'mangaaoonga': 'mangwaaonga',
    'paaoonga':    'paaonga',
    'aaoongi':     'aaongi',
    'jaaoongi':    'jaongi',
    'karoongi':    'karungi',
    'mainne':      'maine',
    'kar raha':    'kar raha',
    'kar rahi':    'kar rahi',
    'jaao':        'jao',
    'aao':         'aao',

    # ── Phrases / religious expressions ─────────────────────────────────
    'meharabani':  'meharbani',
    'haafiz':      'hafiz',
    'alhamdulillaah':'Alhamdulillah',
    'alhamdullah': 'Alhamdulillah',
    'maasha':      'Masha',
    'insha':       'Insha',
    'bismillaah':  'Bismillah',
    'subhaanallaah':'SubhanAllah',
    'astaghfirullaah':'Astaghfirullah',

    # ── Office / business English loans ─────────────────────────────────
    'projekt':     'project',
    'teem':        'team',
    'bos':         'boss',
    'klaaint':     'client',
    'prezenteshan':'presentation',
    'intarawyoo':  'interview',
    'sailri':      'salary',
    'bonas':       'bonus',
    'kanpani':     'company',
    'bizanes':     'business',
    'staartaap':   'startup',
    'maarketing':  'marketing',
    'sels':        'sales',
    'bajat':       'budget',
    'prophit':     'profit',
    'meeting':     'meeting',

    # ── Tech / internet English loans ───────────────────────────────────
    'intaranet':   'internet',
    'waaeephaai':  'wifi',
    'mobaail':     'mobile',
    'laipatop':    'laptop',
    'kanpyootar':  'computer',
    'skreen':      'screen',
    'keebord':     'keyboard',
    'chaarjar':    'charger',
    'baitri':      'battery',
    'richaarj':    'recharge',
    'deta':        'data',
    'kibord':      'keyboard',
    'amaaunt':     'amount',
    'haai':        'high',
    'lo':          'low',
    'paasaward':   'password',
    'aanlaaeen':   'online',
    'aaphalaaeen': 'offline',
    'oke':         'okay',
    'okay':        'okay',

    # ── Compound words — split with space (Roman Urdu convention) ────────
    # Time + bhar (whole/throughout)
    'dinabhar':    'din bhar',     # दिनभर — all day
    'dinbhar':     'din bhar',
    'raatabhar':   'raat bhar',    # रातभर — all night
    'raatbhar':    'raat bhar',
    'saalabhar':   'saal bhar',    # सालभर — all year
    'saalbhar':    'saal bhar',
    'haphtebhar':  'hafte bhar',   # हफ्तेभर — all week
    'mahinebhar':  'mahine bhar',
    'umrabhar':    'umar bhar',    # उम्रभर — lifelong

    # Compound nouns (commonly joined in Devanagari, split in Roman Urdu)
    'haathapair':  'haath paer',   # हाथपैर — hands & feet
    'haathpair':   'haath paer',
    'maanbaap':    'maa baap',     # मांबाप — parents
    'maabaap':     'maa baap',
    'bhaaeebahan': 'bhai behan',   # भाईबहन — brothers & sisters
    'bhaaibahan':  'bhai behan',
    'betaabeti':   'beta beti',    # बेटाबेटी — sons & daughters

    # Compound medical (already partially handled)
    'bladapreshar':'blood pressure',  # ब्लडप्रेशर
    'bladpreshar': 'blood pressure',
    'bladpressure':'blood pressure',
    'dilakeebeemari':'dil ki bimari', # दिलकीबीमारी
    'sheetabukhaar':'sheet bukhaar',  # शीतबुखार
    'khaansijukam':'khansi zukam',    # खांसीज़ुकाम
    'galekharaash':'gale kharaash',   # गलेखराश

    # Directional compounds
    'oopaneeche':  'oopar neeche',
    'aagepeeche':  'aage peechhe',
    'idharaudhar': 'idhar udhar',
    'yahaanwahaan':'yahan wahan',

    # ── Weather ──────────────────────────────────────────────────────────
    'barph':       'barf',          # बर्फ — snow/ice
    'toophaan':    'toofan',        # तूफान — storm
    'kohra':       'kohra',
    'aandhi':      'aandhi',

    # ── Clothing (English loans + Urdu shortening) ───────────────────────
    'shart':       'shirt',
    'paint':       'pant',
    'shalawaar':   'shalwar',
    'taai':        'tie',
    'belt':        'belt',
    'kapre':       'kapre',
    'rumaal':      'rumaal',

    # ── Transport ────────────────────────────────────────────────────────
    'gari':        'gaadi',         # गाड़ी — vehicle
    'bas':         'bus',           # बस — bus (overrides 'bas'=enough; risk noted)
    'tren':        'train',
    'hawaai':      'hawai',         # हवाई — air (long ā shortening)
    'riksha':      'rickshaw',
    'motarasaaikil':'motorcycle',
    'saaikil':     'cycle',
    'steshan':     'station',
    'eyaraport':   'airport',
    'raasta':      'rasta',         # रास्ता — way/road
    'chauk':       'chowk',         # चौक — intersection
    'tikat':       'ticket',
    'seet':        'seat',
    'jahaaz':      'jahaaz',
    'paidal':      'paidal',

    # ── Shopping ────────────────────────────────────────────────────────
    'baazaar':     'bazaar',        # बाज़ार
    'mol':         'mall',
    'diskaaunt':   'discount',
    'ophar':       'offer',
    'sel':         'sale',
    'paik':        'pack',
    'boks':        'box',
    'baig':        'bag',
    'raseed':      'raseed',
    'jhola':       'jhola',
    'dukaan':      'dukaan',
    'haalmaark':   'hallmark',

    # ── Education ───────────────────────────────────────────────────────
    'skool':       'school',
    'kolej':       'college',
    'yooniwarsiti':'university',
    'klaas':       'class',
    'stoodent':    'student',
    'homawark':    'homework',
    'egzaam':      'exam',
    'pepar':       'paper',
    'kitaab':      'kitab',
    'pensil':      'pencil',
    'laaibreri':   'library',
    'prinsipal':   'principal',
    'klaasameet':  'classmate',
    'gradeshan':   'graduation',
    'degri':       'degree',

    # ── Sports / recreation ─────────────────────────────────────────────
    'kriket':      'cricket',
    'phutabol':    'football',
    'hoki':        'hockey',
    'baidamintan': 'badminton',
    'tenis':       'tennis',
    'gem':         'game',
    'maich':       'match',
    'skor':        'score',
    'wiket':       'wicket',
    'philm':       'film',
    'myoozik':     'music',
    'daans':       'dance',
    'paarti':      'party',
    'pikanik':     'picnic',
    'gem oowar':   'game over',

    # ── Religion / festivals (additional + corrected) ───────────────────
    'juma':        'jumma',
    'quraan':      'Quran',
    'maulwi':      'maulvi',
    'imaam':       'imam',
    'hadees':      'hadees',
    'qaazi':       'qazi',
    'charch':      'church',
    'sheikh':      'sheikh',
    'maulaana':    'maulana',
    'haaji':       'haji',
    'tasbih':      'tasbih',
    'zikr':        'zikr',
    'iftaar':      'iftar',
    'sehri':       'sehri',
    'fitra':       'fitra',
    'zakaat':      'zakaat',

    # ── Past tense + double-n corrections ───────────────────────────────
    'hamne':       'humne',
    'unhonne':     'unhone',
    'inhonne':     'inhone',
    'kisine':      'kisi ne',
    'maine':       'maine',     # already
    'tumne':       'tumne',     # already

    # ── Future tense (more) ──────────────────────────────────────────────
    # 'aonga' → 'aaonga' already in dict; verify works

    # ── Urdu religious phrases (capitalisation) ─────────────────────────
    'aameen':      'Ameen',
    'jazaak':      'Jazak',
    'baarak':      'Barak',

    # ── Symptoms (additional) ───────────────────────────────────────────
    'lali':        'laali',         # लाली — redness (single l is wrong)
    'maror':       'marod',         # मरोड़ — cramp (ड़ should be 'd' here colloquially)
    'soojan':      'soojan',
    'khujli':      'khujli',
    'phunsi':      'phunsi',
    'phora':       'phora',
    'zakhm':       'zakhm',
    'chot':        'chot',
    'tees':        'tees',

    # ── Misc adjustments ────────────────────────────────────────────────
    'dhanyawaad':  'dhanyawad',
    'tabiyat':     'tabiyat',
    'sehat':       'sehat',
    'andaaza':     'andaza',
    'taklif':      'taklif',
    'paresh':      'pareshan',     # incomplete word fallback

    # ── Lab test names (compound words & profiles) ──────────────────────
    'lipid':       'lipid',
    'prophaail':   'profile',
    'prophail':    'profile',
    'profile':     'profile',
    'lipid prophaail': 'lipid profile',
    'thyroid prophaail':'thyroid profile',
    'raindam':     'random',
    'random':      'random',
    'peepi':       'PP',           # post-prandial
    'fasting':     'fasting',
    'pp':          'PP',
    'heemoglobin': 'hemoglobin',
    'pletalets':   'platelets',
    'whaait sale': 'white cell',
    'whaait':      'white',
    'red sale':    'red cell',
    'sale':        'sale',         # might conflict — re-add
    'cell':        'cell',
    'grup':        'group',
    'group':       'group',

    # ── Test names (extended) ───────────────────────────────────────────
    'urine rooteen':'urine routine',
    'rooteen':     'routine',
    'urin':        'urine',        # already, but for compound use
    'preganensi':  'pregnancy',
    'pregnensi':   'pregnancy',
    'beeta':       'beta',
    'yoorik':      'uric',
    'esid':        'acid',
    'yoorik esid': 'uric acid',
    'ilektrolaaits':'electrolytes',
    'esajeeoti':   'SGOT',         # fallback if not caught by PROPER_NOUNS
    'esajeepeeti': 'SGPT',
    'echaseeji':   'HCG',

    # ── Imaging / radiology ─────────────────────────────────────────────
    'eks re':      'X-ray',
    'eksre':       'X-ray',
    'altraasaaund':'ultrasound',
    'eeko':        'echo',
    'enjiyograaphi':'angiography',
    'maimograam':  'mammogram',
    'bon densiti': 'bone density',
    'bon':         'bone',
    'densiti':     'density',
    'endoskopi':   'endoscopy',
    'kolonoskopi': 'colonoscopy',

    # ── Infectious diseases ─────────────────────────────────────────────
    'kowid':       'COVID',
    'enteejan':    'antigen',
    'antigen':     'antigen',
    'enteebodi':   'antibody',
    'antibody':    'antibody',
    'dengoo':      'dengue',
    'maleriya':    'malaria',
    'taaiphaaid':  'typhoid',
    'tayphaaid':   'typhoid',
    'widaal':      'Widal',
    'hepatitis':   'hepatitis',
    'kowid test':  'COVID test',
    'kowid peeseeaar':'COVID PCR',

    # ── Specialists (English -ologist suffix) ───────────────────────────
    'kaardiyolojist':'cardiologist',
    'endokraainolojist':'endocrinologist',
    'nephrolojist':'nephrologist',
    'gaistroentrolojist':'gastroenterologist',
    'gaainokolojist':'gynecologist',
    'gynekolojist':'gynecologist',
    'peediyaatrishiyan':'pediatrician',
    'darmatolojist':'dermatologist',
    'nyoorolojist':'neurologist',
    'onkolojist':  'oncologist',
    'yoorolojist': 'urologist',
    'orthopeedik': 'orthopedic',
    'phizeeshiyan':'physician',
    'sarjan':      'surgeon',
    'dentist':     'dentist',
    'saaikiyaatrist':'psychiatrist',
    'raidiyolojist':'radiologist',
    'pathalojist':'pathologist',

    # ── Reasons for tests / categories ──────────────────────────────────
    'enual':       'annual',
    'annual':      'annual',
    'rooteen checkup':'routine checkup',
    'weeza':       'visa',
    'medikal':     'medical',
    'pri':         'pre',
    'pri employament':'pre-employment',
    'employament': 'employment',
    'pri mairej':  'pre-marriage',
    'mairej':      'marriage',
    'inshyorens':  'insurance',
    'helth':       'health',
    'helth check': 'health check',
    'pholo':       'follow',
    'pholo up':    'follow up',
    'konsalteshan':'consultation',
    'kanphidenshal':'confidential',
    'imrejensi':   'emergency',
    'eemarjency':  'emergency',
    'imarjensi':   'emergency',

    # ── Payment / pricing (more) ────────────────────────────────────────
    'ret':         'rate',
    'paikej':      'package',
    'onalaain':    'online',
    'onlaain':     'online',
    'eezeepaisa':  'EasyPaisa',     # also in PROPER_NOUNS for cap
    'jaizakaish':  'JazzCash',
    'repharal':    'referral',
    'referral':    'referral',
    'panel':       'panel',
    'korporet':    'corporate',
    'subsidi':     'subsidy',

    # ── Reports / delivery ──────────────────────────────────────────────
    'haard':       'hard',
    'sopht':       'soft',
    'haard copy':  'hard copy',
    'sopht copy':  'soft copy',
    'whaatsaep':   'WhatsApp',
    'whaatsaep par':'WhatsApp par',
    'peedeeeph':   'PDF',
    'pidiephi':    'PDF',
    'portal':      'portal',
    'mil gai':     'mil gai',
    'mil gaya':    'mil gaya',

    # ── Patient info ────────────────────────────────────────────────────
    'umr':         'umar',
    'umra':        'umar',
    'waalid':      'walid',
    'ilaka':       'ilaqa',
    'makaan':      'makaan',
    'kunaan':      'koonayn',
    'cnic':        'CNIC',
    'siene':       'CNIC',

    # ── Logistics (more) ────────────────────────────────────────────────
    'home sarwis': 'home service',
    'sarwis':      'service',
    'phleebotomist':'phlebotomist',
    'phlebotomist':'phlebotomist',
    'teknishiyan': 'technician',
    'technishiyan':'technician',
    'lab teknishiyan':'lab technician',
    'ayenge':      'aayenge',      # कब आएँगे — when will (they) come
    'aayenge':     'aayenge',
    'aaengey':     'aayenge',
    'ja sakte':    'ja sakte',

    # ── Fasting / preparation (more) ────────────────────────────────────
    'khali':       'khaali',
    'khali pet':   'khaali pet',
    'brash':       'brush',
    'roza':        'roza',
    'naashta':     'naashta',

    # ── Symptoms (more from complaints) ─────────────────────────────────
    'nind':        'neend',
    'bhookh':      'bhookh',
    'wazan':       'wazan',
    'sugar barh gai':'sugar barh gai',
    'barh gai':    'barh gai',
    'barh raha':   'barh raha',
    'kam ho raha': 'kam ho raha',

    # ── Specialty test fragments ────────────────────────────────────────
    'paip smiyar': 'Pap smear',
    'paipsmiyar':  'Pap smear',
    'paip':        'Pap',
    'smiyar':      'smear',
    'rapid test':  'rapid test',
    'aagri':       'angry',

    # ── Multi-word phrases (keys with spaces — handled by phrase pass) ───
    # Format: raw phonetic phrase → corrected form (hyphens, caps, etc.)
    'eks re':            'X-ray',
    'eksare':            'X-ray',
    'pre employment':    'pre-employment',
    'pre marriage':      'pre-marriage',
    'pri employament':   'pre-employment',
    'pri mairej':        'pre-marriage',
    'hepatitis si':      'hepatitis C',
    'hepatitis bi':      'hepatitis B',
    'hepatitis e':       'hepatitis A',
    'vitamin di':        'vitamin D',
    'vitamin bi':        'vitamin B',
    'vitamin si':        'vitamin C',
    'vitamin e':         'vitamin E',
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
    'phaatima':    'Fatima',     # फातिमा without nuqta on फ
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

    # ── Acronyms (need explicit capitalisation) ─────────────────────────
    'aaeedi':      'ID',
    'pin':         'PIN',
    'eteeem':      'ATM',
    'piincode':    'PIN code',
    'seebeesee':   'CBC',
    'elaphti':     'LFT',
    'aaraphti':    'RFT',
    'eemaaraaee':  'MRI',
    'eesheejee':   'ECG',
    'eeeejee':     'EEG',
    'tibi':        'TB',
    'aaeevi':      'IV',
    'eedee':       'AD',
    'pisiar':      'PCR',
    'sionemkri':   'CT',
    'sndi':        'STD',
    'aaivi':       'IV',
    'hpv':         'HPV',
    'edisepi':     'ESR',
    'lit':         'LDL',
    'edichali':    'HDL',

    # ── More lab test acronyms ──────────────────────────────────────────
    'echabeee1si': 'HbA1c',
    'esjeeoti':    'SGOT',
    'esjeepeeti':  'SGPT',
    'teeesaech':   'TSH',
    'tee3':        'T3',
    'tee4':        'T4',
    'echaseeji':   'HCG',
    'peeesae':     'PSA',
    'beepi':       'BP',
    'teebi':       'TB',
    'esaemaes':    'SMS',

    # ── Pakistani cities ────────────────────────────────────────────────
    'laahaur':     'Lahore',
    'peshaawar':   'Peshawar',
    'kweta':       'Quetta',
    'raawalapindi':'Rawalpindi',
    'rawalapindi': 'Rawalpindi',
    'multaan':     'Multan',
    'phaisalaabaad':'Faisalabad',
    'haidaraabaad':'Hyderabad',
    'siyaalkot':   'Sialkot',
    'gujraat':     'Gujrat',
    'sargodha':    'Sargodha',
    'baahaawalpur':'Bahawalpur',

    # ── Indian cities ───────────────────────────────────────────────────
    'munbai':      'Mumbai',
    'kalakatta':   'Kolkata',
    'kolkata':     'Kolkata',
    'chennai':     'Chennai',
    'bangalor':    'Bangalore',
    'haidaraabaad':'Hyderabad',

    # ── Countries ───────────────────────────────────────────────────────
    'indiya':      'India',
    'amerika':     'America',
    'kanada':      'Canada',
    'landan':      'London',
    'dubai':       'Dubai',
    'saoodi':      'Saudi',
    'engaland':    'England',
    'ostreliya':   'Australia',
    'jarmani':     'Germany',
    'phraans':     'France',
    'jaapaan':     'Japan',
    'cheen':       'China',

    # ── Tech brand names ────────────────────────────────────────────────
    'aaeephon':    'iPhone',
    'aiphon':      'iPhone',
    'endroid':     'Android',
    'googal':      'Google',
    'phesabuk':    'Facebook',
    'instaagraam': 'Instagram',
    'yootyoob':    'YouTube',
    'teleegraam':  'Telegram',
    'zoom':        'Zoom',
    'maaikrosoft': 'Microsoft',
    'aiapal':      'Apple',
    'samsang':     'Samsung',

    # ── Banks / orgs ────────────────────────────────────────────────────
    'chughtaaee':  'Chughtai',
    'chughtai':    'Chughtai',
    'esso':        'SSO',
    'naadra':      'NADRA',

    # ── Festivals (caps required) ───────────────────────────────────────
    'holi':        'Holi',
    'diwali':      'Diwali',
    'krisamas':    'Christmas',
    'navratri':    'Navratri',
    'rakhi':       'Rakhi',
    'eed':         'Eid',
    'ramazaan':    'Ramzan',
    'meelaad':     'Milad',

    # ── More acronyms ───────────────────────────────────────────────────
    'aaivee':      'IV',
    'oti':         'OT',
    'aaisiyoo':    'ICU',
    'eemarjency':  'emergency',
    'opd':         'OPD',
    'eepiji':      'APGI',
    'sti':         'STI',
    'std':         'STD',
    'hivi':        'HIV',
    'aaids':       'AIDS',
    'kovid':       'COVID',
    'pisior':      'PCR',
    'rapid':       'rapid',
    'aaiq':        'IQ',

    # ── Tech brands (more) ──────────────────────────────────────────────
    'twitar':      'Twitter',
    'snaepchat':   'Snapchat',
    'linkdin':     'LinkedIn',
    'aaolu':       'Olu',
    'oolu':        'Olu',
    'oochalu':     'Olu',
    'pintarest':   'Pinterest',
    'gmaail':      'Gmail',
    'aautlook':    'Outlook',
    'yaahoo':      'Yahoo',
    'binga':       'Bing',

    # ── Lab test acronyms (Chughtai Lab specific) ───────────────────────
    'seebeesi':    'CBC',
    'sibisee':     'CBC',
    'cbc':         'CBC',
    'elaephti':    'LFT',
    'lft':         'LFT',
    'aaraephti':   'RFT',
    'rft':         'RFT',
    'keephti':     'KFT',
    'kft':         'KFT',
    'echbi':       'Hb',
    'hb':          'Hb',
    'echbi a one si':'HbA1c',
    'echaaaeewi':  'HIV',
    'hiv':         'HIV',
    'aids':        'AIDS',
    'eeseeji':     'ECG',
    'emaaaraaai':  'MRI',
    'eemaaraaee':  'MRI',
    'mri':         'MRI',
    'seeti':       'CT',
    'seeti scan':  'CT scan',
    'esajeeoti':   'SGOT',
    'esajeepeeti': 'SGPT',
    'sgot':        'SGOT',
    'sgpt':        'SGPT',
    'peeseeaar':   'PCR',
    'psa':         'PSA',
    'tsh':         'TSH',
    'hcg':         'HCG',
    'beeta echaseeji':'beta HCG',
    'kovid':       'COVID',
    'covid':       'COVID',
    'di':          'D',           # used with vitamin D
    'bi':          'B',           # used with vitamin B
    'bee12':       'B12',
    'b12':         'B12',
    'opd':         'OPD',
    'ipd':         'IPD',
    'icu':         'ICU',
    'er':          'ER',
    'cnic':        'CNIC',

    # ── Pakistani fintech / brand names ─────────────────────────────────
    'eezeepaisa':  'EasyPaisa',
    'easypaisa':   'EasyPaisa',
    'jaizakaish':  'JazzCash',
    'jazzcash':    'JazzCash',
    'sadapay':     'SadaPay',
    'nayapay':     'NayaPay',
    'jazz':        'Jazz',
    'zong':        'Zong',
    'telenor':     'Telenor',
    'ufone':       'Ufone',
    'paktel':      'Paktel',
    'mobilink':    'Mobilink',

    # ── Imaging / Diagnostic procedure acronyms ─────────────────────────
    'eeeeji':      'EEG',
    'eeji':        'EG',
    'eelteepi':    'LTP',
    'paiseepeesee':'PCP',
    'aiiv':        'IV',

}
