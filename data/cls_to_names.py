import json

flower102_maps = {
    "1": "pink primrose",
    "2": "hard-leaved pocket orchid",
    "3": "canterbury bells",
    "4": "sweet pea",
    "5": "english marigold",
    "6": "tiger lily",
    "7": "moon orchid",
    "8": "bird of paradise",
    "9": "monkshood",
    "10": "globe thistle",
    "11": "snapdragon",
    "12": "colt's foot",
    "13": "king protea",
    "14": "spear thistle",
    "15": "yellow iris",
    "16": "globe flower",
    "17": "purple coneflower",
    "18": "peruvian lily",
    "19": "balloon flower",
    "20": "giant white arum lily",
    "21": "fire lily",
    "22": "pincushion flower",
    "23": "fritillary",
    "24": "red ginger",
    "25": "grape hyacinth",
    "26": "corn poppy",
    "27": "prince of wales feathers",
    "28": "stemless gentian",
    "29": "artichoke",
    "30": "sweet william",
    "31": "carnation",
    "32": "garden phlox",
    "33": "love in the mist",
    "34": "mexican aster",
    "35": "alpine sea holly",
    "36": "ruby-lipped cattleya",
    "37": "cape flower",
    "38": "great masterwort",
    "39": "siam tulip",
    "40": "lenten rose",
    "41": "barbeton daisy",
    "42": "daffodil",
    "43": "sword lily",
    "44": "poinsettia",
    "45": "bolero deep blue",
    "46": "wallflower",
    "47": "marigold",
    "48": "buttercup",
    "49": "oxeye daisy",
    "50": "common dandelion",
    "51": "petunia",
    "52": "wild pansy",
    "53": "primula",
    "54": "sunflower",
    "55": "pelargonium",
    "56": "bishop of llandaff",
    "57": "gaura",
    "58": "geranium",
    "59": "orange dahlia",
    "60": "pink and yellow dahlia",
    "61": "cautleya spicata",
    "62": "japanese anemone",
    "63": "black-eyed susan",
    "64": "silverbush",
    "65": "californian poppy",
    "66": "osteospermum",
    "67": "spring crocus",
    "68": "bearded iris",
    "69": "windflower",
    "70": "tree poppy",
    "71": "gazania",
    "72": "azalea",
    "73": "water lily",
    "74": "rose",
    "75": "thorn apple",
    "76": "morning glory",
    "77": "passion flower",
    "78": "lotus",
    "79": "toad lily",
    "80": "anthurium",
    "81": "frangipani",
    "82": "clematis",
    "83": "hibiscus",
    "84": "columbine",
    "85": "desert-rose",
    "86": "tree mallow",
    "87": "magnolia",
    "88": "cyclamen",
    "89": "watercress",
    "90": "canna lily",
    "91": "hippeastrum",
    "92": "bee balm",
    "93": "air plant",
    "94": "foxglove",
    "95": "bougainvillea",
    "96": "camellia",
    "97": "mallow",
    "98": "mexican petunia",
    "99": "bromelia",
    "100": "blanket flower",
    "101": "trumpet creeper",
    "102": "blackberry lily"
}

flower102_maps = {int(k)-1: v for k, v in flower102_maps.items()}
flower102_maps = dict(sorted(flower102_maps.items()))
flower102_classes = list(flower102_maps.values())

# list of classnames for food101 from the TPT repository
# food101_classes = ['apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare','beet_salad','beignets','bibimbap','bread_pudding','breakfast_burrito','bruschetta','caesar_salad','cannoli','caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi',
#                    'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']

# from the MaPLe repository
food101_classes = ['apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare','beet_salad','beignets','bibimbap','bread_pudding','breakfast_burrito','bruschetta','caesar_salad','cannoli','caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi',
                   'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']

dtd_classes = ['banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 'crosshatched', 'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed', 'interlaced', 'knitted', 'lacelike', 'lined',
               'marbled', 'matted', 'meshed', 'paisley', 'perforated', 'pitted', 'pleated', 'polka-dotted', 'porous', 'potholed', 'scaly', 'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified', 'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged']

pets_classes = ['Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'Bengal', 'Birman', 'Bombay', 'boxer', 'British_Shorthair', 'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin',
                'keeshond', 'leonberger', 'Maine_Coon', 'miniature_pinscher', 'newfoundland', 'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']

sun397_classes = ['abbey', 'airplane cabin', 'airport terminal', 'alley', 'amphitheater', 'amusement arcade', 'amusement park', 'anechoic chamber', 'apartment building (outdoor)', 'apse (indoor)', 'aquarium', 'aqueduct', 'arch', 'archive', 'arrival gate (outdoor)', 'art gallery', 'art school', 'art studio', 'assembly line', 'athletic field (outdoor)', 'atrium (public)', 'attic', 'auditorium', 'auto factory', 'badlands', 'badminton court (indoor)', 'baggage claim', 'bakery (shop)', 'balcony (exterior)', 'balcony (interior)', 'ball pit', 'ballroom', 'bamboo forest', 'banquet hall', 'bar', 'barn', 'barndoor', 'baseball field', 'basement', 'basilica', 'basketball court (outdoor)', 'bathroom', 'batters box', 'bayou', 'bazaar (indoor)', 'bazaar (outdoor)', 'beach', 'beauty salon', 'bedroom', 'berth', 'biology laboratory', 'bistro (indoor)', 'boardwalk', 'boat deck', 'boathouse', 'bookstore', 'booth (indoor)', 'botanical garden', 'bow window (indoor)', 'bow window (outdoor)', 'bowling alley', 'boxing ring', 'brewery (indoor)', 'bridge', 'building facade', 'bullring', 'burial chamber', 'bus interior', 'butchers shop', 'butte', 'cabin (outdoor)', 'cafeteria', 'campsite', 'campus', 'canal (natural)', 'canal (urban)', 'candy store', 'canyon', 'car interior (backseat)', 'car interior (frontseat)', 'carrousel', 'casino (indoor)', 'castle', 'catacomb', 'cathedral (indoor)', 'cathedral (outdoor)', 'cavern (indoor)', 'cemetery', 'chalet', 'cheese factory', 'chemistry lab', 'chicken coop (indoor)', 'chicken coop (outdoor)', 'childs room', 'church (indoor)', 'church (outdoor)', 'classroom', 'clean room', 'cliff', 'cloister (indoor)', 'closet', 'clothing store', 'coast', 'cockpit', 'coffee shop', 'computer room', 'conference center', 'conference room', 'construction site', 'control room', 'control tower (outdoor)', 'corn field', 'corral', 'corridor', 'cottage garden', 'courthouse', 'courtroom', 'courtyard', 'covered bridge (exterior)', 'creek', 'crevasse', 'crosswalk', 'cubicle (office)', 'dam', 'delicatessen', 'dentists office', 'desert (sand)', 'desert (vegetation)', 'diner (indoor)', 'diner (outdoor)', 'dinette (home)', 'dinette (vehicle)', 'dining car', 'dining room', 'discotheque', 'dock', 'doorway (outdoor)', 'dorm room', 'driveway', 'driving range (outdoor)', 'drugstore', 'electrical substation', 'elevator (door)', 'elevator (interior)', 'elevator shaft', 'engine room', 'escalator (indoor)', 'excavation', 'factory (indoor)', 'fairway', 'fastfood restaurant', 'field (cultivated)', 'field (wild)', 'fire escape', 'fire station', 'firing range (indoor)', 'fishpond', 'florist shop (indoor)', 'food court', 'forest (broadleaf)', 'forest (needleleaf)', 'forest path', 'forest road', 'formal garden', 'fountain', 'galley', 'game room', 'garage (indoor)', 'garbage dump', 'gas station', 'gazebo (exterior)', 'general store (indoor)', 'general store (outdoor)', 'gift shop', 'golf course', 'greenhouse (indoor)', 'greenhouse (outdoor)', 'gymnasium (indoor)', 'hangar (indoor)', 'hangar (outdoor)', 'harbor', 
                    'hayfield', 'heliport', 'herb garden', 'highway', 'hill', 'home office', 'hospital', 'hospital room', 'hot spring', 'hot tub (outdoor)', 'hotel (outdoor)', 'hotel room', 'house', 'hunting lodge (outdoor)', 'ice cream parlor', 'ice floe', 'ice shelf', 'ice skating rink (indoor)', 'ice skating rink (outdoor)', 'iceberg', 'igloo', 'industrial area', 'inn (outdoor)', 'islet', 'jacuzzi (indoor)', 'jail (indoor)', 'jail cell', 'jewelry shop', 'kasbah', 'kennel (indoor)', 'kennel (outdoor)', 'kindergarden classroom', 'kitchen', 'kitchenette', 'labyrinth (outdoor)', 'lake (natural)', 'landfill', 'landing deck', 'laundromat', 'lecture room', 'library (indoor)', 'library (outdoor)', 'lido deck (outdoor)', 'lift bridge', 'lighthouse', 'limousine interior', 'living room', 'lobby', 'lock chamber', 'locker room', 'mansion', 'manufactured home', 'market (indoor)', 'market (outdoor)', 'marsh', 'martial arts gym', 'mausoleum', 'medina', 'moat (water)', 'monastery (outdoor)', 'mosque (indoor)', 'mosque (outdoor)', 'motel', 'mountain', 'mountain snowy', 'movie theater (indoor)', 'museum (indoor)', 'music store', 'music studio', 'nuclear power plant (outdoor)', 'nursery', 'oast house', 'observatory (outdoor)', 'ocean', 'office', 'office building', 'oil refinery (outdoor)', 'oilrig', 'operating room', 'orchard', 'outhouse (outdoor)', 'pagoda', 'palace', 'pantry', 'park', 'parking garage (indoor)', 'parking garage (outdoor)', 'parking lot', 'parlor', 'pasture', 'patio', 'pavilion', 'pharmacy', 'phone booth', 'physics laboratory', 'picnic area', 'pilothouse (indoor)', 'planetarium (outdoor)', 'playground', 'playroom', 'plaza', 'podium (indoor)', 'podium (outdoor)', 'pond', 'poolroom (establishment)', 'poolroom (home)', 'power plant (outdoor)', 'promenade deck', 'pub (indoor)', 'pulpit', 'putting green', 'racecourse', 'raceway', 'raft', 'railroad track', 'rainforest', 'reception', 'recreation room', 'residential neighborhood', 'restaurant', 'restaurant kitchen', 'restaurant patio', 'rice paddy', 'riding arena', 'river', 'rock arch', 'rope bridge', 'ruin', 'runway', 'sandbar', 'sandbox', 'sauna', 'schoolhouse', 'sea cliff', 'server room', 'shed', 'shoe shop', 'shopfront', 'shopping mall (indoor)', 'shower', 'skatepark', 'ski lodge', 'ski resort', 'ski slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'squash court', 'stable', 'stadium (baseball)', 'stadium (football)', 'stage (indoor)', 'staircase', 'street', 'subway interior', 'subway station (platform)', 'supermarket', 'sushi bar', 'swamp', 'swimming pool (indoor)', 'swimming pool (outdoor)', 'synagogue (indoor)', 'synagogue (outdoor)', 'television studio', 'temple (east asia)', 'temple (south asia)', 'tennis court (indoor)', 'tennis court (outdoor)', 'tent (outdoor)', 'theater (indoor procenium)', 'theater (indoor seats)', 'thriftshop', 'throne room', 'ticket booth', 'toll plaza', 'topiary garden', 'tower', 'toyshop', 'track (outdoor)', 'train railway', 'train station (platform)', 'tree farm', 'tree house', 'trench', 'underwater (coral reef)', 
                    'utility room', 'valley', 'van interior', 'vegetable garden', 'veranda', 'veterinarians office', 'viaduct', 'videostore', 'village', 'vineyard', 'volcano', 'volleyball court (indoor)', 'volleyball court (outdoor)', 'waiting room', 'warehouse (indoor)', 'water tower', 'waterfall (block)', 'waterfall (fan)', 'waterfall (plunge)', 'watering hole', 'wave', 'wet bar', 'wheat field', 'wind farm', 'windmill', 'wine cellar (barrel storage)', 'wine cellar (bottle storage)', 'wrestling ring (indoor)', 'yard', 'youth hostel']

caltech101_classes = ['face@', 'leopard', 'motorbike', 'accordion', 'airplane', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone',
                      'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']

cars_classes = ['AM General Hummer SUV 2000', 'Acura RL Sedan 2012', 'Acura TL Sedan 2012', 'Acura TL Type-S 2008', 'Acura TSX Sedan 2012', 'Acura Integra Type R 2001', 'Acura ZDX Hatchback 2012', 'Aston Martin V8 Vantage Convertible 2012', 'Aston Martin V8 Vantage Coupe 2012', 'Aston Martin Virage Convertible 2012', 'Aston Martin Virage Coupe 2012', 'Audi RS 4 Convertible 2008', 'Audi A5 Coupe 2012', 'Audi TTS Coupe 2012', 'Audi R8 Coupe 2012', 'Audi V8 Sedan 1994', 'Audi 100 Sedan 1994', 'Audi 100 Wagon 1994', 'Audi TT Hatchback 2011', 'Audi S6 Sedan 2011', 'Audi S5 Convertible 2012', 'Audi S5 Coupe 2012', 'Audi S4 Sedan 2012', 'Audi S4 Sedan 2007', 'Audi TT RS Coupe 2012', 'BMW ActiveHybrid 5 Sedan 2012', 'BMW 1 Series Convertible 2012', 'BMW 1 Series Coupe 2012', 'BMW 3 Series Sedan 2012', 'BMW 3 Series Wagon 2012', 'BMW 6 Series Convertible 2007', 'BMW X5 SUV 2007', 'BMW X6 SUV 2012', 'BMW M3 Coupe 2012', 'BMW M5 Sedan 2010', 'BMW M6 Convertible 2010', 'BMW X3 SUV 2012', 'BMW Z4 Convertible 2012', 'Bentley Continental Supersports Conv. Convertible 2012', 'Bentley Arnage Sedan 2009', 'Bentley Mulsanne Sedan 2011', 'Bentley Continental GT Coupe 2012', 'Bentley Continental GT Coupe 2007', 'Bentley Continental Flying Spur Sedan 2007', 'Bugatti Veyron 16.4 Convertible 2009', 'Bugatti Veyron 16.4 Coupe 2009', 'Buick Regal GS 2012', 'Buick Rainier SUV 2007', 'Buick Verano Sedan 2012', 'Buick Enclave SUV 2012', 'Cadillac CTS-V Sedan 2012', 'Cadillac SRX SUV 2012', 'Cadillac Escalade EXT Crew Cab 2007', 'Chevrolet Silverado 1500 Hybrid Crew Cab 2012', 'Chevrolet Corvette Convertible 2012', 'Chevrolet Corvette ZR1 2012', 'Chevrolet Corvette Ron Fellows Edition Z06 2007', 'Chevrolet Traverse SUV 2012', 'Chevrolet Camaro Convertible 2012', 'Chevrolet HHR SS 2010', 'Chevrolet Impala Sedan 2007', 'Chevrolet Tahoe Hybrid SUV 2012', 'Chevrolet Sonic Sedan 2012', 'Chevrolet Express Cargo Van 2007', 'Chevrolet Avalanche Crew Cab 2012', 'Chevrolet Cobalt SS 2010', 'Chevrolet Malibu Hybrid Sedan 2010', 'Chevrolet TrailBlazer SS 2009', 'Chevrolet Silverado 2500HD Regular Cab 2012', 'Chevrolet Silverado 1500 Classic Extended Cab 2007', 'Chevrolet Express Van 2007', 'Chevrolet Monte Carlo Coupe 2007', 'Chevrolet Malibu Sedan 2007', 'Chevrolet Silverado 1500 Extended Cab 2012', 'Chevrolet Silverado 1500 Regular Cab 2012', 'Chrysler Aspen SUV 2009', 'Chrysler Sebring Convertible 2010', 'Chrysler Town and Country Minivan 2012', 'Chrysler 300 SRT-8 2010', 'Chrysler Crossfire Convertible 2008', 'Chrysler PT Cruiser Convertible 2008', 'Daewoo Nubira Wagon 2002', 'Dodge Caliber Wagon 2012', 'Dodge Caliber Wagon 2007', 'Dodge Caravan Minivan 1997', 'Dodge Ram Pickup 3500 Crew Cab 2010', 'Dodge Ram Pickup 3500 Quad Cab 2009', 'Dodge Sprinter Cargo Van 2009', 'Dodge Journey SUV 2012', 'Dodge Dakota Crew Cab 2010', 'Dodge Dakota Club Cab 2007', 'Dodge Magnum Wagon 2008', 'Dodge Challenger SRT8 2011', 'Dodge Durango SUV 2012', 'Dodge Durango SUV 2007', 'Dodge Charger Sedan 2012', 'Dodge Charger SRT-8 2009', 
                'Eagle Talon Hatchback 1998', 'FIAT 500 Abarth 2012', 'FIAT 500 Convertible 2012', 'Ferrari FF Coupe 2012', 'Ferrari California Convertible 2012', 'Ferrari 458 Italia Convertible 2012', 'Ferrari 458 Italia Coupe 2012', 'Fisker Karma Sedan 2012', 'Ford F-450 Super Duty Crew Cab 2012', 'Ford Mustang Convertible 2007', 'Ford Freestar Minivan 2007', 'Ford Expedition EL SUV 2009', 'Ford Edge SUV 2012', 'Ford Ranger SuperCab 2011', 'Ford GT Coupe 2006', 'Ford F-150 Regular Cab 2012', 'Ford F-150 Regular Cab 2007', 'Ford Focus Sedan 2007', 'Ford E-Series Wagon Van 2012', 'Ford Fiesta Sedan 2012', 'GMC Terrain SUV 2012', 'GMC Savana Van 2012', 'GMC Yukon Hybrid SUV 2012', 'GMC Acadia SUV 2012', 'GMC Canyon Extended Cab 2012', 'Geo Metro Convertible 1993', 'HUMMER H3T Crew Cab 2010', 'HUMMER H2 SUT Crew Cab 2009', 'Honda Odyssey Minivan 2012', 'Honda Odyssey Minivan 2007', 'Honda Accord Coupe 2012', 'Honda Accord Sedan 2012', 'Hyundai Veloster Hatchback 2012', 'Hyundai Santa Fe SUV 2012', 'Hyundai Tucson SUV 2012', 'Hyundai Veracruz SUV 2012', 'Hyundai Sonata Hybrid Sedan 2012', 'Hyundai Elantra Sedan 2007', 'Hyundai Accent Sedan 2012', 'Hyundai Genesis Sedan 2012', 'Hyundai Sonata Sedan 2012', 'Hyundai Elantra Touring Hatchback 2012', 'Hyundai Azera Sedan 2012', 'Infiniti G Coupe IPL 2012', 'Infiniti QX56 SUV 2011', 'Isuzu Ascender SUV 2008', 'Jaguar XK XKR 2012', 'Jeep Patriot SUV 2012', 'Jeep Wrangler SUV 2012', 'Jeep Liberty SUV 2012', 'Jeep Grand Cherokee SUV 2012', 'Jeep Compass SUV 2012', 'Lamborghini Reventon Coupe 2008', 'Lamborghini Aventador Coupe 2012', 'Lamborghini Gallardo LP 570-4 Superleggera 2012', 'Lamborghini Diablo Coupe 2001', 'Land Rover Range Rover SUV 2012', 'Land Rover LR2 SUV 2012', 'Lincoln Town Car Sedan 2011', 'MINI Cooper Roadster Convertible 2012', 'Maybach Landaulet Convertible 2012', 'Mazda Tribute SUV 2011', 'McLaren MP4-12C Coupe 2012', 'Mercedes-Benz 300-Class Convertible 1993', 'Mercedes-Benz C-Class Sedan 2012', 'Mercedes-Benz SL-Class Coupe 2009', 'Mercedes-Benz E-Class Sedan 2012', 'Mercedes-Benz S-Class Sedan 2012', 'Mercedes-Benz Sprinter Van 2012', 'Mitsubishi Lancer Sedan 2012', 'Nissan Leaf Hatchback 2012', 'Nissan NV Passenger Van 2012', 'Nissan Juke Hatchback 2012', 'Nissan 240SX Coupe 1998', 'Plymouth Neon Coupe 1999', 'Porsche Panamera Sedan 2012', 'Ram CV Cargo Van Minivan 2012', 'Rolls-Royce Phantom Drophead Coupe Convertible 2012', 'Rolls-Royce Ghost Sedan 2012', 'Rolls-Royce Phantom Sedan 2012', 'Scion xD Hatchback 2012', 'Spyker C8 Convertible 2009', 'Spyker C8 Coupe 2009', 'Suzuki Aerio Sedan 2007', 'Suzuki Kizashi Sedan 2012', 'Suzuki SX4 Hatchback 2012', 'Suzuki SX4 Sedan 2012', 'Tesla Model S Sedan 2012', 'Toyota Sequoia SUV 2012', 'Toyota Camry Sedan 2012', 'Toyota Corolla Sedan 2012', 'Toyota 4Runner SUV 2012', 'Volkswagen Golf Hatchback 2012', 'Volkswagen Golf Hatchback 1991', 'Volkswagen Beetle Hatchback 2012', 'Volvo C30 Hatchback 2012', 'Volvo 240 Sedan 1993', 'Volvo XC90 SUV 2007', 'smart fortwo Convertible 2012']

ucf101_classes = ['Apply_Eye_Makeup', 'Apply_Lipstick', 'Archery', 'Baby_Crawling', 'Balance_Beam', 'Band_Marching', 'Baseball_Pitch', 'Basketball', 'Basketball_Dunk', 'Bench_Press', 'Biking', 'Billiards', 'Blow_Dry_Hair', 'Blowing_Candles', 'Body_Weight_Squats', 'Bowling', 'Boxing_Punching_Bag', 'Boxing_Speed_Bag', 'Breast_Stroke', 'Brushing_Teeth', 'Clean_And_Jerk', 'Cliff_Diving', 'Cricket_Bowling', 'Cricket_Shot', 'Cutting_In_Kitchen', 'Diving', 'Drumming', 'Fencing', 'Field_Hockey_Penalty', 'Floor_Gymnastics', 'Frisbee_Catch', 'Front_Crawl', 'Golf_Swing', 'Haircut', 'Hammering', 'Hammer_Throw', 'Hand Stand Pushups', 'Handstand_Walking', 'Head_Massage', 'High_Jump', 'Horse_Race', 'Horse_Riding', 'Hula_Hoop', 'Ice_Dancing', 'Javelin_Throw', 'Juggling_Balls', 'Jumping_Jack', 'Jump_Rope',
                  'Kayaking', 'Knitting', 'Long_Jump', 'Lunges', 'Military_Parade', 'Mixing', 'Mopping_Floor', 'Nunchucks', 'Parallel_Bars', 'Pizza_Tossing', 'Playing_Cello', 'Playing_Daf', 'Playing_Dhol', 'Playing_Flute', 'Playing_Guitar', 'Playing_Piano', 'Playing_Sitar', 'Playing_Tabla', 'Playing_Violin', 'Pole_Vault', 'Pommel_Horse', 'Pull_Ups', 'Punch', 'Push_Ups', 'Rafting', 'Rock_Climbing_Indoor', 'Rope_Climbing', 'Rowing', 'Salsa_Spin', 'Shaving_Beard', 'Shotput', 'Skate_Boarding', 'Skiing', 'Skijet', 'Sky_Diving', 'Soccer_Juggling', 'Soccer_Penalty', 'Still_Rings', 'Sumo_Wrestling', 'Surfing', 'Swing', 'Table_Tennis_Shot', 'Tai_Chi', 'Tennis_Swing', 'Throw_Discus', 'Trampoline_Jumping', 'Typing', 'Uneven_Bars', 'Volleyball_Spiking', 'Walking_With_Dog', 'Wall_Pushups', 'Writing_On_Board', 'Yo_Yo']

aircraft_classes = ['707-320', '727-200', '737-200', '737-300', '737-400', '737-500', '737-600', '737-700', '737-800', '737-900', '747-100', '747-200', '747-300', '747-400', '757-200', '757-300', '767-200', '767-300', '767-400', '777-200', '777-300', 'A300B4', 'A310', 'A318', 'A319', 'A320', 'A321', 'A330-200', 'A330-300', 'A340-200', 'A340-300', 'A340-500', 'A340-600', 'A380', 'ATR-42', 'ATR-72', 'An-12', 'BAE 146-200', 'BAE 146-300', 'BAE-125', 'Beechcraft 1900', 'Boeing 717', 'C-130', 'C-47', 'CRJ-200', 'CRJ-700', 'CRJ-900', 'Cessna 172', 'Cessna 208', 'Cessna 525', 'Cessna 560',
                    'Challenger 600', 'DC-10', 'DC-3', 'DC-6', 'DC-8', 'DC-9-30', 'DH-82', 'DHC-1', 'DHC-6', 'DHC-8-100', 'DHC-8-300', 'DR-400', 'Dornier 328', 'E-170', 'E-190', 'E-195', 'EMB-120', 'ERJ 135', 'ERJ 145', 'Embraer Legacy 600', 'Eurofighter Typhoon', 'F-16A/B', 'F/A-18', 'Falcon 2000', 'Falcon 900', 'Fokker 100', 'Fokker 50', 'Fokker 70', 'Global Express', 'Gulfstream IV', 'Gulfstream V', 'Hawk T1', 'Il-76', 'L-1011', 'MD-11', 'MD-80', 'MD-87', 'MD-90', 'Metroliner', 'Model B200', 'PA-28', 'SR-20', 'Saab 2000', 'Saab 340', 'Spitfire', 'Tornado', 'Tu-134', 'Tu-154', 'Yak-42']

eurosat_classes = ['Annual Crop Land', 'Forest', 'Herbaceous Vegetation Land', 'Highway or Road',
                   'Industrial Buildings', 'Pasture Land', 'Permanent Crop Land', 'Residential Buildings', 'River', 'Sea or Lake']

cifar_10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

cifar_100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak', 'orange', 'orchid', 'otter', 'palm', 'pear', 'pickup_truck',
    'pine', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon',
    'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
    'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
    'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow', 'wolf',
    'woman', 'worm'
]
