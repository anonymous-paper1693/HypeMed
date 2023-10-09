
def first_level_code_category():
    """
    根据头一级(3位数字或字母+三位数字)代码进行分类,生成一个字典
    参考:http://www.icd9data.com/2015/Volume1/default.htm
    Returns:

    """
    first_category = {}
    for i in range(1, 1000):
        first_level_code = f'{i:0>3d}'
        if 1 <= i <= 139:
            new_code = "A"  # Infectious And Parasitic Diseases
            if 1 <= i <= 9:
                new_code += "A"  # Intestinal Infectious Diseases
            elif 10 <= i <= 18:
                new_code += "B"  # Tuberculosis
            elif 20 <= i <= 27:
                new_code += "C"  # Zoonotic Bacterial Diseases
            elif 30 <= i <= 41:
                new_code += "D"  # Other Bacterial Diseases
            elif i == 42:
                new_code += "E"  # Human Immunodeficiency Virus
            elif 45 <= i <= 49:
                new_code += "F"  # Poliomyelitis And Other Non-Arthropod-Borne Viral Diseases Of Central Nervous System
            elif 50 <= i <= 59:
                new_code += "G"  # Viral Diseases Accompanied By Exanthem
            elif 60 <= i <= 66:
                new_code += "H"  # Arthropod-Borne Viral Diseases
            elif 70 <= i <= 79:
                new_code += "I"  # Other Diseases Due To Viruses And Chlamydiae
            elif 80 <= i <= 88:
                new_code += "J"  # Rickettsioses And Other Arthropod-Borne Diseases
            elif 90 <= i <= 99:
                new_code += "K"  # Syphilis And Other Venereal Diseases
            elif 100 <= i <= 104:
                new_code += "L"  # Other Spirochetal Diseases
            elif 110 <= i <= 118:
                new_code += "M"  # Mycoses
            elif 120 <= i <= 129:
                new_code += "N"  # Helminthiases
            elif 130 <= i <= 136:
                new_code += "O"  # Other Infectious And Parasitic Diseases
            elif 137 <= i <= 139:
                new_code += "P"  # Late Effects Of Infectious And Parasitic Diseases
        elif 140 <= i <= 239:
            new_code = "B"  # Neoplasms
            if 140 <= i <= 149:
                new_code += "A"  # Malignant Neoplasm Of Lip, Oral Cavity, And Pharynx
            elif 150 <= i <= 159:
                new_code += "B"  # Malignant Neoplasm Of Digestive Organs And Peritoneum
            elif 160 <= i <= 165:
                new_code += "C"  # Malignant Neoplasm Of Respiratory And Intrathoracic Organs
            elif 170 <= i <= 176:
                new_code += "D"  # Malignant Neoplasm Of Bone, Connective Tissue, Skin, And Breast
            elif 179 <= i <= 189:
                new_code += "E"  # Malignant Neoplasm Of Genitourinary Organs
            elif 190 <= i <= 199:
                new_code += "F"  # Malignant Neoplasm Of Other And Unspecified Sites
            elif 200 <= i <= 209:
                new_code += "G"  # Malignant Neoplasm Of Lymphatic And Hematopoietic Tissue
            elif 210 <= i <= 229:
                new_code += "H"  # Benign Neoplasms
            elif 230 <= i <= 234:
                new_code += "I"  # Carcinoma In Situ
            elif 235 <= i <= 238:
                new_code += "J"  # Neoplasms Of Uncertain Behavior
            elif i == 239:
                new_code += "K"  # Neoplasms Of Unspecified Nature
        elif 240 <= i <= 279:
            new_code = "C"  # Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders
            if 240 <= i <= 246:
                new_code += "A"  # Disorders Of Thyroid Gland
            elif 249 <= i <= 259:
                new_code += "B"  # Diseases Of Other Endocrine Glands
            elif 260 <= i <= 269:
                new_code += "C"  # Nutritional Deficiencies
            elif 270 <= i <= 279:
                new_code += "D"  # Other Metabolic Disorders And Immunity Disorders
        elif 280 <= i <= 289:
            new_code = "D"  # Diseases Of The Blood And Blood-Forming Organs
            if 280 <= i <= 280:
                new_code += "A"  # Iron deficiency anemias
            elif 281 <= i <= 281:
                new_code += "B"  # Other deficiency anemias
            elif 282 <= i <= 282:
                new_code += "C"  # Hereditary hemolytic anemias
            elif 283 <= i <= 283:
                new_code += "D"  # Acquired hemolytic anemias
            elif 284 <= i <= 284:
                new_code += "E"  # Aplastic anemia and other bone marrow failure syndromes
            elif 285 <= i <= 285:
                new_code += "F"  # Other and unspecified anemias
            elif 286 <= i <= 286:
                new_code += "G"  # Coagulation defects
            elif 287 <= i <= 287:
                new_code += "H"  # Purpura and other hemorrhagic conditions
            elif 288 <= i <= 288:
                new_code += "I"  # Diseases of white blood cells
            elif 289 <= i <= 289:
                new_code += "J"  # Other diseases of blood and blood-forming organs
        elif 290 <= i <= 319:
            new_code = "E"  # Mental Disorders
            if 290 <= i <= 294:
                new_code += "A"  # Organic Psychotic Conditions
            elif 295 <= i <= 299:
                new_code += "B"  # Other Psychoses
            elif 300 <= i <= 316:
                new_code += "C"  # Neurotic Disorders, Personality Disorders, And Other Nonpsychotic Mental Disorders
            elif 317 <= i <= 319:
                new_code += "D"  # Intellectual Disabilities
        elif 320 <= i <= 389:
            new_code = "F"  # Diseases Of The Nervous System And Sense Organs
            if 320 <= i <= 327:
                new_code += "A"  # Inflammatory Diseases Of The Central Nervous System
            elif 330 <= i <= 337:
                new_code += "B"  # Hereditary And Degenerative Diseases Of The Central Nervous System
            elif i == 338:
                new_code += "C"  # Pain
            elif i == 339:
                new_code += "D"  # Other Headache Syndromes
            elif 340 <= i <= 349:
                new_code += "E"  # Other Disorders Of The Central Nervous System
            elif 350 <= i <= 359:
                new_code += "F"  # Disorders Of The Peripheral Nervous System
            elif 360 <= i <= 379:
                new_code += "G"  # Disorders Of The Eye And Adnexa
            elif 380 <= i <= 389:
                new_code += "H"  # Diseases Of The Ear And Mastoid Process
        elif 390 <= i <= 459:
            new_code = "G"  # Diseases Of The Circulatory System
            if 390 <= i <= 392:
                new_code += "A"  # Acute Rheumatic Fever
            elif 393 <= i <= 398:
                new_code += "B"  # Chronic Rheumatic Heart Disease
            elif 401 <= i <= 405:
                new_code += "C"  # Hypertensive Disease
            elif 410 <= i <= 414:
                new_code += "D"  # Ischemic Heart Disease
            elif 415 <= i <= 417:
                new_code += "E"  # Diseases Of Pulmonary Circulation
            elif 420 <= i <= 429:
                new_code += "F"  # Other Forms Of Heart Disease
            elif 430 <= i <= 438:
                new_code += "G"  # Cerebrovascular Disease
            elif 440 <= i <= 449:
                new_code += "H"  # Diseases Of Arteries, Arterioles, And Capillaries
            elif 451 <= i <= 459:
                new_code += "I"  # Diseases Of Veins And Lymphatics, And Other Diseases Of Circulatory System
        elif 460 <= i <= 519:
            new_code = "H"  # Diseases Of The Respiratory System
            if 460 <= i <= 466:
                new_code += "A"  # Acute Respiratory Infections
            elif 470 <= i <= 478:
                new_code += "B"  # Other Diseases Of Upper Respiratory Tract
            elif 480 <= i <= 488:
                new_code += "C"  # Pneumonia And Influenza
            elif 490 <= i <= 496:
                new_code += "D"  # Chronic Obstructive Pulmonary Disease And Allied Conditions
            elif 500 <= i <= 508:
                new_code += "E"  # Pneumoconioses And Other Lung Diseases Due To External Agents
            elif 510 <= i <= 519:
                new_code += "F"  # Other Diseases Of Respiratory System
        elif 520 <= i <= 579:
            new_code = "I"  # Diseases Of The Digestive System
            if 520 <= i <= 529:
                new_code += "A"  # Diseases Of Oral Cavity, Salivary Glands, And Jaws
            elif 530 <= i <= 539:
                new_code += "B"  # Diseases Of Esophagus, Stomach, And Duodenum
            elif 540 <= i <= 543:
                new_code += "C"  # Appendicitis
            elif 550 <= i <= 553:
                new_code += "D"  # Hernia Of Abdominal Cavity
            elif 555 <= i <= 558:
                new_code += "E"  # Noninfective Enteritis And Colitis
            elif 560 <= i <= 569:
                new_code += "F"  # Other Diseases Of Intestines And Peritoneum
            elif 570 <= i <= 579:
                new_code += "G"  # Other Diseases Of Digestive System
        elif 580 <= i <= 629:
            new_code = "J"  # Diseases Of The Genitourinary System
            if 580 <= i <= 589:
                new_code += "A"  # Nephritis, Nephrotic Syndrome, And Nephrosis
            elif 590 <= i <= 599:
                new_code += "B"  # Other Diseases Of Urinary System
            elif 600 <= i <= 608:
                new_code += "C"  # Diseases Of Male Genital Organs
            elif 610 <= i <= 612:
                new_code += "D"  # Disorders Of Breast
            elif 614 <= i <= 616:
                new_code += "E"  # Inflammatory Disease Of Female Pelvic Organs
            elif 617 <= i <= 629:
                new_code += "F"  # Other Disorders Of Female Genital Tract
        elif 630 <= i <= 679:
            new_code = "K"  # Complications Of Pregnancy, Childbirth, And The Puerperium
            if 630 <= i <= 639:
                new_code += "A"  # Ectopic And Molar Pregnancy And Other Pregnancy With Abortive Outcome
            elif 640 <= i <= 649:
                new_code += "B"  # Complications Mainly Related To Pregnancy
            elif 650 <= i <= 659:
                new_code += "C"  # Normal Delivery, And Other Indications For Care In Pregnancy, Labor, And Delivery
            elif 660 <= i <= 669:
                new_code += "D"  # Complications Occurring Mainly In The Course Of Labor And Delivery
            elif 670 <= i <= 677:
                new_code += "E"  # Complications Of The Puerperium
            elif 678 <= i <= 679:
                new_code += "F"  # Other Maternal And Fetal Complications
        elif 680 <= i <= 709:
            new_code = "L"  # Diseases Of The Skin And Subcutaneous Tissue
            if 680 <= i <= 686:
                new_code += "A"  # Infections Of Skin And Subcutaneous Tissue
            elif 690 <= i <= 698:
                new_code += "B"  # Other Inflammatory Conditions Of Skin And Subcutaneous Tissue
            elif 700 <= i <= 709:
                new_code += "C"  # Other Diseases Of Skin And Subcutaneous Tissue
        elif 710 <= i <= 739:
            new_code = "M"  # Diseases Of The Musculoskeletal System And Connective Tissue
            if 710 <= i <= 719:
                new_code += "A"  # Arthropathies And Related Disorders
            elif 720 <= i <= 724:
                new_code += "B"  # Dorsopathies
            elif 725 <= i <= 729:
                new_code += "C"  # Rheumatism, Excluding The Back
            elif 730 <= i <= 739:
                new_code += "D"  # Osteopathies, Chondropathies, And Acquired Musculoskeletal Deformities
        elif 740 <= i <= 759:
            new_code = "N"  # Congenital Anomalies
            if 740 <= i <= 749:
                new_code += "A"  # Anencephalus and similar anomalies
            elif i == 741:
                new_code += "B"  # Spina bifida
            elif i == 742:
                new_code += "C"  # Other congenital anomalies of nervous system
            elif i == 743:
                new_code += "D"  # Congenital anomalies of eye
            elif i == 744:
                new_code += "E"  # Congenital anomalies of ear face and neck
            elif 745 <= i <= 746:
                new_code += "F"  # Bulbus cordis anomalies and anomalies of cardiac septal closure, Other congenital anomalies of heart
            elif i == 747:
                new_code += "G"  # Other congenital anomalies of circulatory system
            elif i == 748:
                new_code += "H"  # Congenital anomalies of respiratory system
            elif i == 749:
                new_code += "I"  # Cleft palate and cleft lip
            elif 750 <= i <= 751:
                new_code += "J"  # Other congenital anomalies of upper alimentary tract, Other congenital anomalies of digestive system
            elif i == 752:
                new_code += "K"  # Congenital anomalies of genital organs
            elif i == 753:
                new_code += "L"  # Congenital anomalies of urinary system
            elif i == 754:
                new_code += "M"  # Certain congenital musculoskeletal deformities
            elif 755 <= i <= 756:
                new_code += "N"  # Other congenital anomalies of limbs, Other congenital musculoskeletal anomalies
            elif i == 757:
                new_code += "O"  # Congenital anomalies of the integument
            elif i == 758:
                new_code += "P"  # Chromosomal anomalies
            elif i == 759:
                new_code += "Q"  # Other and unspecified congenital anomalies
        elif 760 <= i <= 779:
            new_code = "O"  # Certain Conditions Originating In The Perinatal Period
            if 760 <= i <= 763:
                new_code += "A"  # Maternal Causes Of Perinatal Morbidity And Mortality
            elif 764 <= i <= 779:
                new_code += "B"  # Other Conditions Originating In The Perinatal Period
        elif 780 <= i <= 799:
            new_code = "P"  # Symptoms, Signs, And Ill-Defined Conditions
            if 780 <= i <= 789:
                new_code += "A"  # Symptoms
            elif 790 <= i <= 796:
                new_code += "B"  # Nonspecific Abnormal Findings
            elif 797 <= i <= 799:
                new_code += "C"  # Ill-Defined And Unknown Causes Of Morbidity And Mortality
        elif 800 <= i <= 999:
            new_code = "Q"  # Injury And Poisoning
            if 800 <= i <= 804:
                new_code += "A"  # Fracture Of Skull
            elif 805 <= i <= 809:
                new_code += "B"  # Fracture Of Spine And Trunk
            elif 810 <= i <= 819:
                new_code += "C"  # Fracture Of Upper Limb
            elif 820 <= i <= 829:
                new_code += "D"  # Fracture Of Lower Limb
            elif 830 <= i <= 839:
                new_code += "E"  # Dislocation
            elif 840 <= i <= 848:
                new_code += "F"  # Sprains And Strains Of Joints And Adjacent Muscles
            elif 850 <= i <= 854:
                new_code += "G"  # Intracranial Injury, Excluding Those With Skull Fracture
            elif 860 <= i <= 869:
                new_code += "H"  # Internal Injury Of Chest, Abdomen, And Pelvis
            elif 870 <= i <= 879:
                new_code += "I"  # Open Wound Of Head, Neck, And Trunk
            elif 880 <= i <= 887:
                new_code += "J"  # Open Wound Of Upper Limb
            elif 890 <= i <= 897:
                new_code += "K"  # Open Wound Of Lower Limb
            elif 900 <= i <= 904:
                new_code += "L"  # Injury To Blood Vessels
            elif 905 <= i <= 909:
                new_code += "M"  # Late Effects Of Injuries, Poisonings, Toxic Effects, And Other External Causes
            elif 910 <= i <= 919:
                new_code += "N"  # Superficial Injury
            elif 920 <= i <= 924:
                new_code += "O"  # Contusion With Intact Skin Surface
            elif 925 <= i <= 929:
                new_code += "P"  # Crushing Injury
            elif 930 <= i <= 939:
                new_code += "Q"  # Effects Of Foreign Body Entering Through Orifice
            elif 940 <= i <= 949:
                new_code += "R"  # Burns
            elif 950 <= i <= 957:
                new_code += "S"  # Injury To Nerves And Spinal Cord
            elif 958 <= i <= 959:
                new_code += "T"  # Certain Traumatic Complications And Unspecified Injuries
            elif 960 <= i <= 979:
                new_code += "U"  # Poisoning By Drugs, Medicinals And Biological Substances
            elif 980 <= i <= 989:
                new_code += "V"  # Toxic Effects Of Substances Chiefly Nonmedicinal As To Source
            elif 990 <= i <= 995:
                new_code += "W"  # Other And Unspecified Effects Of External Causes
            elif 996 <= i <= 999:
                new_code += "X"  # Complications Of Surgical And Medical Care, Not Elsewhere Classified
        first_category[first_level_code] = new_code
    # 补充E的情况
    for i in range(0, 1000):
        first_level_code = f'E{i:0>3d}'
        new_code = 'R'
        if i == 0:
            new_code += "A"  # External Cause Status
        elif 1 <= i <= 30:
            new_code += "B"  # Activity
        elif 800 <= i <= 807:
            new_code += "C"  # Railway Accidents
        elif 810 <= i <= 819:
            new_code += "D"  # Motor Vehicle Traffic Accidents
        elif 820 <= i <= 825:
            new_code += "E"  # Motor Vehicle Nontraffic Accidents
        elif 826 <= i <= 829:
            new_code += "F"  # Other Road Vehicle Accidents
        elif 830 <= i <= 838:
            new_code += "G"  # Water Transport Accidents
        elif 840 <= i <= 845:
            new_code += "H"  # Air And Space Transport Accidents
        elif 846 <= i <= 849:
            new_code += "I"  # Vehicle Accidents, Not Elsewhere Classifiable
        elif 850 <= i <= 858:
            new_code += "J"  # Accidental Poisoning By Drugs, Medicinal Substances, And Biologicals
        elif 860 <= i <= 869:
            new_code += "K"  # Accidental Poisoning By Other Solid And Liquid Substances, Gases, And Vapors
        elif 870 <= i <= 876:
            new_code += "L"  # Misadventures To Patients During Surgical And Medical Care
        elif 878 <= i <= 879:
            new_code += "M"  # Surgical And Medical Procedures As The Cause Of Abnormal Reaction Of Patient Or Later Complication, Without Mention Of Misadventure At The Time Of Procedure
        elif 880 <= i <= 888:
            new_code += "N"  # Accidental Falls
        elif 890 <= i <= 899:
            new_code += "O"  # Accidents Caused By Fire And Flames
        elif 900 <= i <= 909:
            new_code += "P"  # Accidents Due To Natural And Environmental Factors
        elif 910 <= i <= 915:
            new_code += "Q"  # Accidents Caused By Submersion, Suffocation, And Foreign Bodies
        elif 916 <= i <= 928:
            new_code += "R"  # Other Accidents
        elif i == 929:
            new_code += "S"  # Late Effects Of Accidental Injury
        elif 930 <= i <= 949:
            new_code += "T"  # Drugs, Medicinal And Biological Substances Causing Adverse Effects In Therapeutic Use
        elif 950 <= i <= 959:
            new_code += "U"  # Suicide And Self-Inflicted Injury
        elif 960 <= i <= 969:
            new_code += "V"  # Homicide And Injury Purposely Inflicted By Other Persons
        elif 970 <= i <= 979:
            new_code += "W"  # Legal Intervention
        elif 980 <= i <= 989:
            new_code += "X"  # Injury Undetermined Whether Accidentally Or Purposely Inflicted
        elif 990 <= i <= 999:
            new_code += "Y"  # Injury Resulting From Operations Of War
        first_category[first_level_code] = new_code

    # 补充V的情况
    for i in range(1, 92):
        new_code = 'S'
        first_level_code = f'V{i:0>2d}'
        if 1 <= i <= 9:
            new_code += "A"  # Persons With Potential Health Hazards Related To Communicable Diseases
        elif 10 <= i <= 19:
            new_code += "B"  # Persons With Potential Health Hazards Related To Personal And Family History
        elif 20 <= i <= 29:
            new_code += "C"  # Persons Encountering Health Services In Circumstances Related To Reproduction And Development
        elif 30 <= i <= 39:
            new_code += "D"  # Liveborn Infants According To Type Of Birth
        elif 40 <= i <= 49:
            new_code += "E"  # Persons With A Condition Influencing Their Health Status
        elif 50 <= i <= 59:
            new_code += "F"  # Persons Encountering Health Services For Specific Procedures And Aftercare
        elif 60 <= i <= 69:
            new_code += "G"  # Persons Encountering Health Services In Other Circumstances
        elif 70 <= i <= 82:
            new_code += "H"  # Persons Without Reported Diagnosis Encountered During Examination And Investigation Of Individuals And Populations
        elif 83 <= i <= 84:
            new_code += "I"  # Genetics
        elif i == 85:
            new_code += "J"  # Body Mass Index
        elif i == 86:
            new_code += "K"  # Estrogen Receptor Status
        elif i == 87:
            new_code += "L"  # Other Specified Personal Exposures And History Presenting Hazards To Health
        elif i == 88:
            new_code += "M"  # Acquired Absence Of Other Organs And Tissue
        elif i == 89:
            new_code += "N"  # Other Suspected Conditions Not Found
        elif i == 90:
            new_code += "O"  # Retained Foreign Body
        elif i == 91:
            new_code += "P"  # Multiple Gestation Placenta Status
        first_category[first_level_code] = new_code

    return first_category

