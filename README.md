# Klasifikacija raka kože

## Kako pokrenuti repozitorijum
Da biste pokrenuli repozitorijum, neophodno je pratiti sledeće korake:
1. Preuzeti skup podataka
2. Instalirati Python 3 i njegove neophodne biblioteke
3. Preuzeti postojeće (istrenirane) modele (opciono)
4. Podesiti argumente komandne linije (opciono)
5. Koristeći Python, pokrenuti `main.py`

_Poželjno je imati CUDA uređaj za brže izvršavanje modela (oba modela su optimizovana na rad sa grafičkom kartom)_

## Uputstvo za preuzimanje skupa podataka

Preuzeti [ISIC 2016 Task 3 skup podataka](https://challenge.isic-archive.com/data/#2016) i dekompresovati datoteku u direktorijum `data/dataset`. Ovo ukljucuje `.zip` podatke za trening i test, kao i datoteke koje sadrže labele (eng. ground truth) u `.csv` formatu. <br /><br />
_ISIC 2016 Task 3:_
- [Slike za trening skup](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip)
- [Labelirani trening skup](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv)
- [Slike za test skup](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_Data.zip)
- [Labelirani test skup](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_GroundTruth.csv)

## Neohpodne Python biblioteke

- torch
- torchvision
- numpy
- scikit-learn
- pickle
- matplotlib
- seaborn
- pandas
- PIL

## Preuzimanje modela

Ako ne želite da trenirate modele, postojeće istrenirane modele možete preuzeti preko [ovog linka](https://drive.google.com/drive/folders/1XcwSpf8gvSaUsEOvT9Lm8NGkfIrLn6dV?usp=sharing). Preuzmite ih u direktorijum `models/`.

## Argumenti komandne linije
Prilikom pokretanja `main.py`, korisnik može uneti željena svojstva programa koji će se pokrenuti. Na primer, ako korisnik samo želi da trenira VGG i prikaže rezultate, to može uraditi sa sledećom komandom:
<br />
`python main.py --type TEST --model_name VGG --show_results TRUE`
<br />


<b> --type       </b>   (Ukucati deo klasifikacije koji će se pokrenuti - TRAIN, TEST ili TRAIN_AND_TEST) <br />
<b> --model_name </b>   (Odabrati model koji će se koristiti - VGG ili XGBoost) <br />
<b> --dataset_name </b>   (Odabrati skup podataka koji će se koristiti - ISIC ili HAM) <br />
<b> --show_results </b>   (Prikaži grafike treninga i/ili testa) <br />
<b> --save_results </b>   (Sačuvaj grafike treninga i/ili testa) <br />
<b> --save_model </b>   (Sačuvaj modele tokom treninga) <br />
<b> --log </b>   (Ispis standardnog izlaza u .log fajl) <br />

Informacije se takođe mogu naći unošenjem <b>--help</b> parametra pri pokretanju `main.py`

## Promena parametara

Za napredna podešavanja, konstantni parametri su dati u datoteci `utils/constants.py`. Može se menjati putanja do skupa podataka, broj epoha i mnoge druge stvari.

## Rezultati

Rezultate, na osnovu modela i skupa podataka, možete videti u `results/` direktorijumu. U formatu `.log` je sačuvan ispis modela.

## Testiranje sa HAM10000 skupom podataka

Dodata je podrška za <b>testiranje</b> na HAM10000 skupu podataka. Link za preuzimanje se može naći na sajtu [ovde](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) klikom na dugme <i>Download</i>. Podaci će biti skinuti u `.zip` formatu koje treba dekompresovati i ubaciti u `data/dataset/archive` direktorijum.
<br>
Prilikom pokretanja programa, uneti argument komandne linije: 
<br>
<b><i>--dataset_name HAM</i></b>

<i>Upozorenje: Skup podataka zauzima 6GB slobodnog prostora i nema podršku za trening</i>

## Seminarski rad
Seminarski rad na <b>srpskom</b> se može naći [ovde](documentation/klasifikacija%20raka.pdf)

## Kako program funkcioniše?

Prilikom pokretanja `main.py` datoteke, u zavisnosti od argumenata komandne linije, poziva se trening ili test (ili oba) iz te datoteke. 
### Trening
Kada se pokrene trening u `train.py` datoteci, podaci se smeštaju u DataLoader, klasi u PyTorch biblioteci specijalizovanoj za kreiranje skupa podataka iz sirovih podataka. Ovde se mogu odrediti npr. specijalne transformacije koje želimo da ima naš skup, tj. možemo imati različite transformacije za XGBoost i VGG model.
Nakon treniranja, model se čuva u `models/` direktorijum.
### Test
Kada se pokrene test u `test.py` datoteci, podaci se takođe smeštaju u Dataloader klasu sa izabranim transformacijama. Test datoteka onda u zavisnosti od modela, drugačije evaluira model i pokazuje grafičke rezultate, ali i pamti izlaz (ako je tako podešeno u argumentu komandne linije <i>--log</i>)

## Informacije o autorima i mentoru

Ovaj projekat su radili __Igor Zolotarev__ i __Branko Grbić__, studenti Matematičkog Fakulteta u sklopu projekta za kurs Istraživanje Podataka 2, pod mentorstvom profesora __Nenada Mitića__.
<br />
Ovim putem se zahvaljujemo na svim preporukama i savetima.