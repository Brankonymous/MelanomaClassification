# Klasifikacija Melanoma raka

## Kako pokrenuti repozitorijum
Da biste pokrenuli repozitorijum, neophodno je pratiti sledeće korake:
1. Preuzeti skup podataka
2. Instalirati Python 3 i njegove neophodne biblioteke
3. Podesiti argumente komandne linije (opciono)
4. Koristeći Python, pokrenuti `main.py`

Poželjno je imati CUDA uređaj za brže izvršavanje modela (oba modela su optimizovana na rad sa grafičkom kartom)

## Uputstvo za preuzimanje skupa podataka

Skinuti [ISIC 2016 Task 3 skup podataka](https://challenge.isic-archive.com/data/#2016) u direktorijum `data/dataset`. Ovo ukljucuje `.zip` podatke za trening i test, kao i njihove datoteke koje sadrže labele (eng. ground truth) u `.csv` formatu:
- [Slike za trening skup](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip)
- [Labelirani trening skup](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip)
- [Slike za test skup](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_Data.zip)
- [Labelirani test skup](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip)

## Neohpodne Python biblioteke

- torch
- numpy
- scikit-learn
- pickle
- matplotlib
- seaborn
- torchvision
- pandas
- PIL

## Argumenti komandne linije
Prilikom pokretanja `main.py`, korisnik može uneti željena svojstva programa koji će se pokrenuti. Na primer, ako korisnik samo želi da trenira VGG i prikaže rezultate, to može uraditi sa sledećom komandom:
<br />
`Python main.py --type TEST --model_name VGG --show_results TRUE`
<br />


<b> --type       </b>   (Ukucati deo klasifikacije koji će se pokrenuti - TRAIN, TEST ili TRAIN_AND_TEST) <br />
<b> --model_name </b>   (Odabrati model koji će se koristiti - VGG ili XGBoost) <br />
<b> --show_results </b>   (Prikaži grafike treninga i/ili testa) <br />
<b> --save_results </b>   (Sačuvaj grafike treninga i/ili testa) <br />
<b> --save_model </b>   (Sačuvaj modele tokom treninga) <br />

Informacije se takođe mogu naći unošenjem <b>--help</b> parametra pri pokretanju `main.py`