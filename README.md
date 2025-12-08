# SIM-PUK
- Samia Wahib `psv745`
- Ida Marie Grøn `zpv282`

## data
### Target municipalities
- Albertslund
- Ballerup
- Brøndby
- Dragør
- Frederiksberg
- Furesø
- Gentofte
- Gladsaxe
- Glostrup
- Greve
- Herlev
- Hvidovre
- Ishøj
- København
- Lyngby-Taarbæk
- Rødovre
- Rudersdal
- Tårnby
- Vallensbæk

### data collected for every candidate
Some fields may be blank if the candidate has not filled in the information.
This list of labels is from the column names in the .csv files.
- `id`: candidate ID given by TV2
- `name`: full legal name
- `votes`: number of votes received
- `elected`: a boolean stating wheter or not they got elected
- `age`: the age as of the day we scraped the data
- `party`: the letter representing the party
- `marital status`
- `number of kids`
- `postal number`
- `occupation`: as described by the candidate themselves
- answers to questions
  - answers converted to number (from completely agreee = 0 to completely disagree = 4)
- comments (optional) for every question
- list of their most important themes
- a shorter and a longer pitch for themselves

## be aware of before running
- latest scraping date: 08/12/2025
- the data of number of votes and the elected bool does *disappear* from the pages at times
  - we do not know the pattern, but they always come back at some point
- there is one candidate we cannot get all of the information from 
  - it is Henning Kornbo from Furesø (ID:120497045)
  - the personal site always returns a status code 500
  - as a solution we skip any personal site returning the status code 500
  
### requirements
- beautiful soup 4 (`pip install beautifulsoup4`)
- lxml (`pip install lxml`)