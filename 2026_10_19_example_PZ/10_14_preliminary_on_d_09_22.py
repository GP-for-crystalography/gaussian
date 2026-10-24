# 14.10.2026
# indexing once again on dictionary from 22.09 but using derivative and sinFit

import os, sys
from pathlib import Path
import numpy as np
sys.path.append('C:\\Users\\ITSupport\\Documents\\Dominik\\Niobe_II_kod\\pychord') #path do katalogu z pychord
# sys.path.append('C:\\Users\\domin\\OneDrive - Uniwersytet Jagielloński\\Studia - SMP - semestr III\\Tutoring - Niobe II\\Symulacja - EMSoft - VS friendly\\pychord')
import EMSoftHandling as EM 
import indexingMTX as id
EM.emSoftPATH = 'C:\\Users\\ITSupport\\Documents\\Dominik\\EMsoft-5.0.20231226.-Win64\\bin'
# ============= Template for copying ================

# dirMain = Path('c:\\Users\\domin\\Documents\\Studia - SMP - semestr 4\\Niobe II - pliki za duze na OneDrive')
# dirTemplates = 
# dirAngles = dirMain/
# dirDictionary = dirMain/
# dirExperimental = dirMain/
# fileExperimentalStack = dirExperimental/
# imageSize =  #lenght of the edge of an image
# nmlFileName = 
# outputFilePrefix = 
# outputDir = dirMain/
# outputDir.mkdir(exist_ok=True)
# outputDirPseudo = outputDir/
# outputDirPseudo.mkdir(exist_ok=True)

# ============= Indexing run KTN 014 ================

dirMain = 
dirTemplates = dirMain/'2026_10_19_example_PZ'
dirAngles = dirTemplates/'angles'
dirDictionary = dirTemplates/'dictionary'
dirDictionary.mkdir(exist_ok=True)
fileExperimentalStack = dirTemplates/'KTN_014_s2_fin_crop_372x372_bin_62x62.tif'
fileTemplate = dirTemplates/'ECP_KTN_014_01_s02_temp.nml'
outputDir = dirTemplates


if __name__ == '__main__':# ważne, bo multiprocessing sie wywali
    EM.mkDictionaryParallel(dirTemplates,
                            fileTemplate,
                            angleFileDir = dirAngles,
                            outputDir = outputDir,
                            coneAngle = 40, #pochył próbki, parametr z pomiarów
                            angleStep = 2, #skok obrotu, parametr z pomiarów
                            )
    id.imageIndexing(dirDictionary,
                    fileExperimentalStack,
                    dirAngles = dirAngles,
                    chunkSize = 100, #ile wektorów eksperymentalnych procesowanych na raz, limit - RAM 
                    imageSize = 64, 
                    outputFilePrefix = 'PZ_test_KTN',
                    outputDir = outputDir,
                    performSineFit = True,
                    derivativeIndexing = True #licz iloczyn skalarny z pochodnej z sygnału
                    )