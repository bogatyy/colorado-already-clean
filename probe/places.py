"""Probe set for the cross-model "Colorado neuron" study.

Every entry is (wikipedia_title, template_name, category, group).
  - wikipedia_title: English Wikipedia article used as a document probe.
  - template_name:   how the place is written inside the short template sentences
                     (None = document probe only, no template).
  - category:        coarse type used for selection / controls.
  - group:           finer label (US state for US entities, otherwise a region tag).
"""

# --- US states: (state, wiki title, capital, second city) -----------------------------
US_STATES = [
    ("Alabama", "Alabama", "Montgomery", "Birmingham"),
    ("Alaska", "Alaska", "Juneau", "Anchorage"),
    ("Arizona", "Arizona", "Phoenix", "Tucson"),
    ("Arkansas", "Arkansas", "Little Rock", "Fayetteville"),
    ("California", "California", "Sacramento", "Los Angeles"),
    ("Colorado", "Colorado", "Denver", "Colorado Springs"),
    ("Connecticut", "Connecticut", "Hartford", "New Haven"),
    ("Delaware", "Delaware", "Dover", "Wilmington"),
    ("Florida", "Florida", "Tallahassee", "Miami"),
    ("Georgia", "Georgia (U.S. state)", "Atlanta", "Savannah"),
    ("Hawaii", "Hawaii", "Honolulu", "Hilo"),
    ("Idaho", "Idaho", "Boise", "Coeur d'Alene"),
    ("Illinois", "Illinois", "Springfield", "Chicago"),
    ("Indiana", "Indiana", "Indianapolis", "Fort Wayne"),
    ("Iowa", "Iowa", "Des Moines", "Cedar Rapids"),
    ("Kansas", "Kansas", "Topeka", "Wichita"),
    ("Kentucky", "Kentucky", "Frankfort", "Louisville"),
    ("Louisiana", "Louisiana", "Baton Rouge", "New Orleans"),
    ("Maine", "Maine", "Augusta", "Portland"),
    ("Maryland", "Maryland", "Annapolis", "Baltimore"),
    ("Massachusetts", "Massachusetts", "Boston", "Worcester"),
    ("Michigan", "Michigan", "Lansing", "Detroit"),
    ("Minnesota", "Minnesota", "Saint Paul", "Minneapolis"),
    ("Mississippi", "Mississippi", "Jackson", "Gulfport"),
    ("Missouri", "Missouri", "Jefferson City", "St. Louis"),
    ("Montana", "Montana", "Helena", "Billings"),
    ("Nebraska", "Nebraska", "Lincoln", "Omaha"),
    ("Nevada", "Nevada", "Carson City", "Las Vegas"),
    ("New Hampshire", "New Hampshire", "Concord", "Manchester"),
    ("New Jersey", "New Jersey", "Trenton", "Newark"),
    ("New Mexico", "New Mexico", "Santa Fe", "Albuquerque"),
    ("New York", "New York (state)", "Albany", "Buffalo"),
    ("North Carolina", "North Carolina", "Raleigh", "Charlotte"),
    ("North Dakota", "North Dakota", "Bismarck", "Fargo"),
    ("Ohio", "Ohio", "Columbus", "Cleveland"),
    ("Oklahoma", "Oklahoma", "Oklahoma City", "Tulsa"),
    ("Oregon", "Oregon", "Salem", "Portland"),
    ("Pennsylvania", "Pennsylvania", "Harrisburg", "Philadelphia"),
    ("Rhode Island", "Rhode Island", "Providence", "Newport"),
    ("South Carolina", "South Carolina", "Columbia", "Charleston"),
    ("South Dakota", "South Dakota", "Pierre", "Sioux Falls"),
    ("Tennessee", "Tennessee", "Nashville", "Memphis"),
    ("Texas", "Texas", "Austin", "Houston"),
    ("Utah", "Utah", "Salt Lake City", "Provo"),
    ("Vermont", "Vermont", "Montpelier", "Burlington"),
    ("Virginia", "Virginia", "Richmond", "Virginia Beach"),
    ("Washington", "Washington (state)", "Olympia", "Seattle"),
    ("West Virginia", "West Virginia", "Charleston", "Morgantown"),
    ("Wisconsin", "Wisconsin", "Madison", "Milwaukee"),
    ("Wyoming", "Wyoming", "Cheyenne", "Jackson"),
]
# Bare city names that are too ambiguous to use as a template name on their own.
AMBIGUOUS_CITY = {
    "Springfield", "Columbus", "Portland", "Jackson", "Lincoln", "Concord", "Manchester",
    "Newark", "Albany", "Columbia", "Charleston", "Pierre", "Augusta", "Dover", "Wilmington",
    "Richmond", "Madison", "Salem", "Austin", "Phoenix", "Helena", "Olympia", "Burlington",
    "Fayetteville", "Birmingham", "Newport", "Providence", "Buffalo", "Montgomery", "Frankfort",
    "Worcester", "Gulfport", "Hartford", "Trenton", "Montpelier",
}

# --- Extra Colorado documents (the "positive" set gets richer coverage) -----------------
COLORADO_EXTRA = [
    ("Boulder, Colorado", "Boulder"), ("Fort Collins, Colorado", "Fort Collins"),
    ("Aurora, Colorado", None), ("Pueblo, Colorado", None),
    ("Grand Junction, Colorado", "Grand Junction"), ("Aspen, Colorado", "Aspen"),
    ("Vail, Colorado", "Vail"), ("Breckenridge, Colorado", "Breckenridge"),
    ("Telluride, Colorado", "Telluride"), ("Durango, Colorado", None),
    ("Pikes Peak", "Pikes Peak"), ("Rocky Mountain National Park", "Rocky Mountain National Park"),
    ("Mesa Verde National Park", "Mesa Verde"), ("Garden of the Gods", None),
    ("Red Rocks Amphitheatre", "Red Rocks"), ("University of Colorado Boulder", None),
    ("Colorado State University", None), ("Denver Broncos", "the Denver Broncos"),
    ("Colorado Rockies", None), ("Denver Nuggets", None), ("Colorado Avalanche", None),
    ("Front Range", "the Front Range"), ("Mount Elbert", "Mount Elbert"),
    ("Great Sand Dunes National Park and Preserve", None), ("History of Colorado", None),
    ("Colorado School of Mines", None), ("United States Air Force Academy", None),
    ("Steamboat Springs, Colorado", "Steamboat Springs"), ("San Juan Mountains", "the San Juan Mountains"),
    ("Geography of Colorado", None),
]

# --- Matched US controls outside Colorado (type-matched to the Colorado extras) -----------
US_OTHER = [
    ("Yellowstone National Park", "Yellowstone", "Wyoming"), ("Yosemite National Park", "Yosemite", "California"),
    ("Grand Canyon", "the Grand Canyon", "Arizona"), ("Glacier National Park (U.S.)", "Glacier National Park", "Montana"),
    ("Grand Teton National Park", "Grand Teton", "Wyoming"), ("Zion National Park", "Zion National Park", "Utah"),
    ("Great Smoky Mountains National Park", "the Great Smoky Mountains", "Tennessee"),
    ("Everglades National Park", "the Everglades", "Florida"), ("Acadia National Park", "Acadia", "Maine"),
    ("Mount Rainier", "Mount Rainier", "Washington"), ("Denali", "Denali", "Alaska"),
    ("Mount Whitney", "Mount Whitney", "California"), ("Mount Washington", "Mount Washington", "New Hampshire"),
    ("Park City, Utah", "Park City", "Utah"), ("Jackson Hole", "Jackson Hole", "Wyoming"),
    ("Lake Tahoe", "Lake Tahoe", "California"), ("Sun Valley, Idaho", "Sun Valley", "Idaho"),
    ("Taos, New Mexico", "Taos", "New Mexico"), ("Stowe, Vermont", "Stowe", "Vermont"),
    ("Big Sky, Montana", "Big Sky", "Montana"), ("Moab, Utah", "Moab", "Utah"),
    ("Flagstaff, Arizona", "Flagstaff", "Arizona"), ("Bozeman, Montana", "Bozeman", "Montana"),
    ("Missoula, Montana", "Missoula", "Montana"), ("Laramie, Wyoming", "Laramie", "Wyoming"),
    ("Asheville, North Carolina", "Asheville", "North Carolina"),
    ("University of Utah", None, "Utah"), ("University of Michigan", None, "Michigan"),
    ("University of Texas at Austin", None, "Texas"), ("University of Washington", None, "Washington"),
    ("University of Wyoming", None, "Wyoming"), ("Montana State University", None, "Montana"),
    ("Ohio State University", None, "Ohio"), ("University of Florida", None, "Florida"),
    ("Georgia Institute of Technology", None, "Georgia"), ("United States Military Academy", None, "New York"),
    ("Utah Jazz", None, "Utah"), ("Dallas Cowboys", "the Dallas Cowboys", "Texas"),
    ("Seattle Seahawks", None, "Washington"), ("Kansas City Chiefs", None, "Missouri"),
    ("Arizona Diamondbacks", None, "Arizona"), ("Minnesota Wild", None, "Minnesota"),
    ("Green Bay Packers", None, "Wisconsin"), ("Boston Celtics", None, "Massachusetts"),
    ("Wasatch Range", "the Wasatch Range", "Utah"), ("Teton Range", "the Teton Range", "Wyoming"),
    ("Sierra Nevada", "the Sierra Nevada", "California"), ("Cascade Range", "the Cascades", "Washington"),
    ("Appalachian Mountains", "the Appalachians", "East"), ("Great Plains", "the Great Plains", "Plains"),
    ("Ozarks", "the Ozarks", "Missouri"), ("Black Hills", "the Black Hills", "South Dakota"),
    ("History of Utah", None, "Utah"), ("History of Texas", None, "Texas"),
    ("Geography of Utah", None, "Utah"), ("Geography of Montana", None, "Montana"),
]

# --- Countries --------------------------------------------------------------------------
# (wiki title, template name, tag) ; tag marks the a-priori hypotheses from the Goodfire map
COUNTRIES = """
Afghanistan|Afghanistan
Albania|Albania
Algeria|Algeria
Andorra|Andorra
Angola|Angola
Argentina|Argentina
Armenia|Armenia
Australia|Australia
Austria|Austria
Azerbaijan|Azerbaijan
The Bahamas|the Bahamas
Bahrain|Bahrain
Bangladesh|Bangladesh
Barbados|Barbados
Belarus|Belarus
Belgium|Belgium
Belize|Belize
Benin|Benin
Bhutan|Bhutan
Bolivia|Bolivia
Bosnia and Herzegovina|Bosnia
Botswana|Botswana
Brazil|Brazil
Brunei|Brunei
Bulgaria|Bulgaria
Burkina Faso|Burkina Faso
Burundi|Burundi
Cambodia|Cambodia
Cameroon|Cameroon
Canada|Canada
Cape Verde|Cape Verde
Central African Republic|the Central African Republic
Chad|Chad
Chile|Chile
China|China
Colombia|Colombia
Comoros|Comoros
Democratic Republic of the Congo|the Democratic Republic of the Congo
Republic of the Congo|the Republic of the Congo
Costa Rica|Costa Rica
Croatia|Croatia
Cuba|Cuba
Cyprus|Cyprus
Czech Republic|the Czech Republic
Denmark|Denmark
Djibouti|Djibouti
Dominican Republic|the Dominican Republic
East Timor|East Timor
Ecuador|Ecuador
Egypt|Egypt
El Salvador|El Salvador
Equatorial Guinea|Equatorial Guinea
Eritrea|Eritrea
Estonia|Estonia
Eswatini|Eswatini
Ethiopia|Ethiopia
Fiji|Fiji
Finland|Finland
France|France
Gabon|Gabon
The Gambia|the Gambia
Georgia (country)|the country of Georgia
Germany|Germany
Ghana|Ghana
Greece|Greece
Guatemala|Guatemala
Guinea|Guinea
Guinea-Bissau|Guinea-Bissau
Guyana|Guyana
Haiti|Haiti
Honduras|Honduras
Hungary|Hungary
Iceland|Iceland
India|India
Indonesia|Indonesia
Iran|Iran
Iraq|Iraq
Republic of Ireland|Ireland
Israel|Israel
Italy|Italy
Ivory Coast|Ivory Coast
Jamaica|Jamaica
Japan|Japan
Jordan|Jordan
Kazakhstan|Kazakhstan
Kenya|Kenya
Kosovo|Kosovo
Kuwait|Kuwait
Kyrgyzstan|Kyrgyzstan
Laos|Laos
Latvia|Latvia
Lebanon|Lebanon
Lesotho|Lesotho
Liberia|Liberia
Libya|Libya
Liechtenstein|Liechtenstein
Lithuania|Lithuania
Luxembourg|Luxembourg
Madagascar|Madagascar
Malawi|Malawi
Malaysia|Malaysia
Maldives|the Maldives
Mali|Mali
Malta|Malta
Mauritania|Mauritania
Mauritius|Mauritius
Mexico|Mexico
Moldova|Moldova
Monaco|Monaco
Mongolia|Mongolia
Montenegro|Montenegro
Morocco|Morocco
Mozambique|Mozambique
Myanmar|Myanmar
Namibia|Namibia
Nepal|Nepal
Netherlands|the Netherlands
New Zealand|New Zealand
Nicaragua|Nicaragua
Niger|Niger
Nigeria|Nigeria
North Korea|North Korea
North Macedonia|North Macedonia
Norway|Norway
Oman|Oman
Pakistan|Pakistan
Panama|Panama
Papua New Guinea|Papua New Guinea
Paraguay|Paraguay
Peru|Peru
Philippines|the Philippines
Poland|Poland
Portugal|Portugal
Qatar|Qatar
Romania|Romania
Russia|Russia
Rwanda|Rwanda
Samoa|Samoa
San Marino|San Marino
Saudi Arabia|Saudi Arabia
Senegal|Senegal
Serbia|Serbia
Seychelles|Seychelles
Sierra Leone|Sierra Leone
Singapore|Singapore
Slovakia|Slovakia
Slovenia|Slovenia
Solomon Islands|the Solomon Islands
Somalia|Somalia
South Africa|South Africa
South Korea|South Korea
South Sudan|South Sudan
Spain|Spain
Sri Lanka|Sri Lanka
Sudan|Sudan
Suriname|Suriname
Sweden|Sweden
Switzerland|Switzerland
Syria|Syria
Taiwan|Taiwan
Tajikistan|Tajikistan
Tanzania|Tanzania
Thailand|Thailand
Togo|Togo
Tonga|Tonga
Trinidad and Tobago|Trinidad and Tobago
Tunisia|Tunisia
Turkey|Turkey
Turkmenistan|Turkmenistan
Uganda|Uganda
Ukraine|Ukraine
United Arab Emirates|the United Arab Emirates
United Kingdom|the United Kingdom
Uruguay|Uruguay
Uzbekistan|Uzbekistan
Vanuatu|Vanuatu
Venezuela|Venezuela
Vietnam|Vietnam
Yemen|Yemen
Zambia|Zambia
Zimbabwe|Zimbabwe
"""

# Landlocked / mountainous annotations used only in the analysis (not in selection).
LANDLOCKED = {
    "Afghanistan", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Bhutan", "Bolivia",
    "Botswana", "Burkina Faso", "Burundi", "Central African Republic", "Chad", "Czech Republic",
    "Eswatini", "Ethiopia", "Hungary", "Kazakhstan", "Kosovo", "Kyrgyzstan", "Laos", "Lesotho",
    "Liechtenstein", "Luxembourg", "Malawi", "Mali", "Moldova", "Mongolia", "Nepal", "Niger",
    "North Macedonia", "Paraguay", "Rwanda", "San Marino", "Serbia", "Slovakia", "South Sudan",
    "Switzerland", "Tajikistan", "Turkmenistan", "Uganda", "Uzbekistan", "Zambia", "Zimbabwe",
}
# Mean elevation of the country above ~1000 m (rough, from standard country-elevation tables).
HIGH_ELEVATION = {
    "Bhutan", "Nepal", "Tajikistan", "Kyrgyzstan", "Lesotho", "Andorra", "Afghanistan", "Chile",
    "China", "Armenia", "Rwanda", "Georgia (country)", "Peru", "Mongolia", "Switzerland", "Ethiopia",
    "Bolivia", "Ecuador", "Iran", "Turkey", "Burundi", "Mexico", "Eswatini", "Namibia", "Zambia",
    "Zimbabwe", "South Africa", "Kenya", "Tanzania", "Yemen", "Lebanon", "Colombia", "Botswana",
    "Angola", "Uganda", "Eritrea", "Austria", "Kosovo", "North Macedonia", "Montenegro", "Azerbaijan",
}

# --- Sub-national regions outside the US ----------------------------------------------
REGIONS = """
Alberta|Alberta|canada
British Columbia|British Columbia|canada
Saskatchewan|Saskatchewan|canada
Manitoba|Manitoba|canada
Ontario|Ontario|canada
Quebec|Quebec|canada
New Brunswick|New Brunswick|canada
Nova Scotia|Nova Scotia|canada
Prince Edward Island|Prince Edward Island|canada
Newfoundland and Labrador|Newfoundland|canada
Yukon|the Yukon|canada
Northwest Territories|the Northwest Territories|canada
Nunavut|Nunavut|canada
Tibet|Tibet|asia
Kashmir|Kashmir|asia
Ladakh|Ladakh|asia
Sikkim|Sikkim|asia
Himachal Pradesh|Himachal Pradesh|asia
Uttarakhand|Uttarakhand|asia
Kerala|Kerala|asia
Punjab|Punjab|asia
Sichuan|Sichuan|asia
Yunnan|Yunnan|asia
Xinjiang|Xinjiang|asia
Qinghai|Qinghai|asia
Inner Mongolia|Inner Mongolia|asia
Hokkaido|Hokkaido|asia
Okinawa Prefecture|Okinawa|asia
Bali|Bali|asia
Siberia|Siberia|asia
Balochistan|Balochistan|asia
Kurdistan|Kurdistan|asia
Tyrol (federal state)|Tyrol|europe
Bavaria|Bavaria|europe
Scotland|Scotland|europe
Wales|Wales|europe
Catalonia|Catalonia|europe
Basque Country (autonomous community)|the Basque Country|europe
Transylvania|Transylvania|europe
Lapland (Finland)|Lapland|europe
Sicily|Sicily|europe
Tuscany|Tuscany|europe
Savoie|Savoie|europe
Valais|Valais|europe
Graubünden|Graubünden|europe
Patagonia|Patagonia|southamerica
Altiplano|the Altiplano|southamerica
La Paz Department (Bolivia)|La Paz Department|southamerica
Potosí Department|Potosí Department|southamerica
Santa Cruz Department (Bolivia)|Santa Cruz Department|southamerica
Cusco Region|the Cusco Region|southamerica
Mendoza Province|Mendoza Province|southamerica
Jujuy Province|Jujuy|southamerica
Atacama Desert|the Atacama Desert|southamerica
Amazonas (Brazilian state)|Amazonas|southamerica
Tasmania|Tasmania|oceania
Queensland|Queensland|oceania
South Island|the South Island|oceania
Greenland|Greenland|other
Puerto Rico|Puerto Rico|other
Baja California|Baja California|other
Chihuahua (state)|Chihuahua|other
Yucatán|Yucatán|other
Western Cape|the Western Cape|africa
KwaZulu-Natal|KwaZulu-Natal|africa
Zanzibar|Zanzibar|africa
"""

# --- Mountain ranges, peaks and landscape features ---------------------------------------
FEATURES = """
Andes|the Andes|mountain
Rocky Mountains|the Rocky Mountains|mountain
Canadian Rockies|the Canadian Rockies|mountain
Alps|the Alps|mountain
Himalayas|the Himalayas|mountain
Hindu Kush|the Hindu Kush|mountain
Karakoram|the Karakoram|mountain
Pamir Mountains|the Pamir Mountains|mountain
Tian Shan|the Tian Shan|mountain
Caucasus Mountains|the Caucasus|mountain
Pyrenees|the Pyrenees|mountain
Carpathian Mountains|the Carpathians|mountain
Atlas Mountains|the Atlas Mountains|mountain
Ural Mountains|the Urals|mountain
Altai Mountains|the Altai Mountains|mountain
Dolomites|the Dolomites|mountain
Scottish Highlands|the Scottish Highlands|mountain
Ethiopian Highlands|the Ethiopian Highlands|mountain
Southern Alps|the Southern Alps|mountain
Drakensberg|the Drakensberg|mountain
Tibetan Plateau|the Tibetan Plateau|mountain
Mount Everest|Mount Everest|peak
Aconcagua|Aconcagua|peak
Mont Blanc|Mont Blanc|peak
Mount Kilimanjaro|Kilimanjaro|peak
Matterhorn|the Matterhorn|peak
K2|K2|peak
Mount Fuji|Mount Fuji|peak
Machu Picchu|Machu Picchu|site
Lake Titicaca|Lake Titicaca|site
Salar de Uyuni|the Salar de Uyuni|site
Amazon rainforest|the Amazon rainforest|lowland
Sahara|the Sahara|lowland
Gobi Desert|the Gobi Desert|lowland
Great Barrier Reef|the Great Barrier Reef|lowland
Mississippi River|the Mississippi River|lowland
Nile|the Nile|lowland
Pampas|the Pampas|lowland
Serengeti|the Serengeti|lowland
Mekong Delta|the Mekong Delta|lowland
Caribbean|the Caribbean|lowland
Colorado Plateau|the Colorado Plateau|coloword
Great Basin|the Great Basin|lowland
"""

# --- World cities -----------------------------------------------------------------------
WORLD_CITIES = """
La Paz|La Paz|Bolivia
Sucre|Sucre|Bolivia
Santa Cruz de la Sierra|Santa Cruz de la Sierra|Bolivia
Cochabamba|Cochabamba|Bolivia
Potosí|Potosí|Bolivia
El Alto|El Alto|Bolivia
Quito|Quito|Ecuador
Cusco|Cusco|Peru
Lima|Lima|Peru
Arequipa|Arequipa|Peru
Bogotá|Bogotá|Colombia
Medellín|Medellín|Colombia
Santiago|Santiago|Chile
Buenos Aires|Buenos Aires|Argentina
Mendoza, Argentina|Mendoza|Argentina
Bariloche|Bariloche|Argentina
Asunción|Asunción|Paraguay
Montevideo|Montevideo|Uruguay
Caracas|Caracas|Venezuela
Rio de Janeiro|Rio de Janeiro|Brazil
São Paulo|São Paulo|Brazil
Kabul|Kabul|Afghanistan
Kandahar|Kandahar|Afghanistan
Kathmandu|Kathmandu|Nepal
Pokhara|Pokhara|Nepal
Thimphu|Thimphu|Bhutan
Lhasa|Lhasa|Tibet
Bern|Bern|Switzerland
Zürich|Zürich|Switzerland
Geneva|Geneva|Switzerland
Zermatt|Zermatt|Switzerland
Davos|Davos|Switzerland
St. Moritz|St. Moritz|Switzerland
Innsbruck|Innsbruck|Austria
Salzburg|Salzburg|Austria
Vienna|Vienna|Austria
Chamonix|Chamonix|France
Grenoble|Grenoble|France
Calgary|Calgary|Canada
Edmonton|Edmonton|Canada
Banff, Alberta|Banff|Canada
Vancouver|Vancouver|Canada
Whistler, British Columbia|Whistler|Canada
Toronto|Toronto|Canada
Montreal|Montreal|Canada
Mexico City|Mexico City|Mexico
Guadalajara|Guadalajara|Mexico
Addis Ababa|Addis Ababa|Ethiopia
Nairobi|Nairobi|Kenya
Kigali|Kigali|Rwanda
Ulaanbaatar|Ulaanbaatar|Mongolia
Bishkek|Bishkek|Kyrgyzstan
Dushanbe|Dushanbe|Tajikistan
Almaty|Almaty|Kazakhstan
Tbilisi|Tbilisi|Georgia
Yerevan|Yerevan|Armenia
Skopje|Skopje|North Macedonia
Andorra la Vella|Andorra la Vella|Andorra
Amsterdam|Amsterdam|Netherlands
Copenhagen|Copenhagen|Denmark
Dhaka|Dhaka|Bangladesh
Singapore|Singapore|Singapore
London|London|United Kingdom
Edinburgh|Edinburgh|United Kingdom
Paris|Paris|France
Tokyo|Tokyo|Japan
Sapporo|Sapporo|Japan
Nagano (city)|Nagano|Japan
Cairo|Cairo|Egypt
Lagos|Lagos|Nigeria
Sydney|Sydney|Australia
Queenstown, New Zealand|Queenstown|New Zealand
Moscow|Moscow|Russia
Beijing|Beijing|China
Chengdu|Chengdu|China
Mumbai|Mumbai|India
Shimla|Shimla|India
Darjeeling|Darjeeling|India
Havana|Havana|Cuba
Reykjavík|Reykjavík|Iceland
Cape Town|Cape Town|South Africa
Johannesburg|Johannesburg|South Africa
Dubai|Dubai|United Arab Emirates
Istanbul|Istanbul|Turkey
Tehran|Tehran|Iran
Rome|Rome|Italy
Turin|Turin|Italy
Berlin|Berlin|Germany
Munich|Munich|Germany
Madrid|Madrid|Spain
Lisbon|Lisbon|Portugal
Prague|Prague|Czech Republic
Warsaw|Warsaw|Poland
Stockholm|Stockholm|Sweden
Oslo|Oslo|Norway
Lillehammer|Lillehammer|Norway
Helsinki|Helsinki|Finland
Dublin|Dublin|Ireland
Bangkok|Bangkok|Thailand
Jakarta|Jakarta|Indonesia
Manila|Manila|Philippines
"""

# --- Other senses of the word "Colorado" (lexical-route controls) --------------------------
COLORADO_WORD = [
    ("Colorado River", "the Colorado River"), ("Colorado Party (Paraguay)", None),
    ("Colorado Party (Uruguay)", None), ("Laguna Colorada", "Laguna Colorada"),
    ("Colorado potato beetle", None), ("Colorado River (Argentina)", None),
    ("Colorado Desert", "the Colorado Desert"), ("Colorado City, Arizona", None),
    ("Colorado River (Texas)", None),
]

# --- Topic documents (document probes only) ----------------------------------------------
TOPICS = [
    "Coca", "Skiing", "Mountaineering", "Altitude sickness", "Gold rush", "Silver mining",
    "Cannabis (drug)", "Craft brewery and microbrewery", "Hiking", "Snowboarding", "Elk",
    "Chronic wasting disease", "Xeriscaping", "Llama", "Quinoa", "Evo Morales", "Inca Empire",
    "Sherpa people", "Taliban", "Landlocked country", "Plateau", "Continental divide", "Ranch",
    "Cowboy", "National park", "Ski resort", "Oil shale", "Uranium mining", "American bison",
    "Rock climbing", "Avalanche", "Glacier", "Tree line", "Alpine climate", "Trout", "Wildfire",
    "Mountain biking", "Whitewater kayaking", "Hot spring", "Ghost town",
]

# --- Generic non-geographic controls -----------------------------------------------------
GENERIC = [
    "Photosynthesis", "Python (programming language)", "French Revolution", "Jazz", "DNA",
    "Black hole", "Impressionism", "Stock market", "Basketball", "Calculus", "Shakespeare",
    "Internal combustion engine", "Buddhism", "Chess", "Vaccine", "World War II", "Opera",
    "Machine learning", "Roman Empire", "Coffee", "Electricity", "Olympic Games", "Novel",
    "Protein", "Plate tectonics", "Linux", "Baroque", "Titanic", "Christmas", "Rainbow",
    "Insulin", "Guitar", "Cryptography", "Democracy", "Inflation", "Evolution", "Aspirin",
    "Photography", "Semiconductor", "Poetry", "Tennis", "Algebra", "Renaissance", "Antibiotic",
    "Climate change", "Telephone", "Hinduism", "Baseball", "Nuclear fission", "Ballet",
    "Operating system", "Marriage", "Chocolate", "Railway", "Philosophy", "Law", "Honey bee",
    "Printing press", "Supreme Court of the United States", "Football", "Solar System",
    "Human brain", "Bread", "Bicycle", "Television", "Banking", "Diabetes", "Wine",
    "Periodic table", "Cinema of the United States", "Mathematics", "Dog", "Cat", "Piano",
    "Constitution", "Agriculture", "Steel", "Cotton", "Sleep", "Memory",
]


def _rows(block):
    out = []
    for line in block.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        out.append(parts)
    return out


def build():
    """Return list of dicts: id, title, name, cat, group."""
    E = []

    def add(title, name, cat, group):
        E.append({"id": len(E), "title": title, "name": name, "cat": cat, "group": group})

    for st, title, cap, city2 in US_STATES:
        add(title, st, "us_state", st)
        for c in (cap, city2):
            wiki = f"{c}, {st}"
            name = None if c in AMBIGUOUS_CITY else c
            add(wiki, name, "us_city", st)
    for title, name in COLORADO_EXTRA:
        add(title, name, "colorado_extra", "Colorado")
    for title, name, st in US_OTHER:
        add(title, name, "us_other", st)
    for title, name in _rows(COUNTRIES):
        add(title, name, "country", title)
    for title, name, grp in _rows(REGIONS):
        add(title, name, "region", grp)
    for title, name, grp in _rows(FEATURES):
        add(title, name, "feature", grp)
    for title, name, grp in _rows(WORLD_CITIES):
        add(title, name, "world_city", grp)
    for title, name in COLORADO_WORD:
        add(title, name, "colorado_word", "coloword")
    for t in TOPICS:
        add(t, None, "topic", "topic")
    for t in GENERIC:
        add(t, None, "generic", "generic")
    return E


TEMPLATES = [
    "I just got back from a two-week trip to {X}, and honestly",
    "My cousin moved to {X} last year. She says the",
    "The weather in {X} this time of year is usually",
    "We spent most of the summer in {X}. Every morning we would",
    "Q: What is {X} best known for?\nA: It is best known for",
    "A few things everyone should know about {X}: first,",
]

if __name__ == "__main__":
    E = build()
    from collections import Counter
    print(len(E), Counter(e["cat"] for e in E))
    print(sum(1 for e in E if e["name"]), "template names")
