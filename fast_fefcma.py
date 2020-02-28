# -*- coding: utf-8 -*-
from imports import *




options = {}
options['N_char'] = 50
options['max_rank'] = 10
options['punct'] = False
options['word'] = False


langA = 'UD_French-GSD'
langB = 'UD_English-EWT'


#Build chars stats
dataA, dataB = getCharsStats(langA), getCharsStats(langB)

#preprocess
dataA['chars'],dataA['data'] = dataA['chars'][0:options['N_char']], dataA['data'][0:options['N_char']]
dataB['chars'],dataB['data'] = dataB['chars'][0:options['N_char']], dataB['data'][0:options['N_char']]
charsA, charsB = dataA['chars'], dataB['chars']
wordsA, wordsB = loadWords(langA), loadWords(langB)

#Détecte si le combo de langue est réaliste par croisements des caractères fréquents
nthres_char_cross = 20
charsA_fsort = list(countChars(wordsA)[0].keys())[0:nthres_char_cross]
charsB_fsort = list(countChars(wordsB)[0].keys())[0:nthres_char_cross]
chars_cross = len([True for c in charsA_fsort if c in charsB_fsort]) / len(set(charsA_fsort+charsB_fsort))
print(f"Chars proximity: {chars_cross*100:.2f}%")

# =============================================================================
# # SWAPCHARS MATRICIEL
# =============================================================================
# création des matrices
langs = [langA, langB]
mats, chars_ranks = build_words_mats(langs)


#Init for rank mapping
crA, crB = chars_ranks[langA], chars_ranks[langB]
rank_to_charsA, rank_to_charsB = {v:k for k,v in crA.items()}, {v:k for k,v in crB.items()}
rank_to_charsA[-1],rank_to_charsA[-2] = '_', ''
rank_to_charsB[-1],rank_to_charsB[-2] = '_', ''


#===
#Algorithme de recherche de mapping
#===
#Init
#bypass_already_mapped = True #définir des intervalles (true sur les rangs inférieur, false après 30)
rank_limit = 20 #définir des intervalles {10:[0,20], 30:[20:60], 50:[20:100]}
maxrange_to_tryA = 40
maxrange_to_tryB = 70 
mapped_scores = {}
len_intval = [6,99]
mapping_limit_per_char = {1:3,3:1} #i_retry1=>2, i_retry2=>2, i_retry3=>1..
n_retry = 5

#calcule la couverture du default
rank_mapping = {i:-1 for i in range(maxrange_to_tryB)}
mat_dict_mapped = create_mapped_mat(mats[langA], rank_mapping)
default_cov_scores = compute_coverages(mats[langB], mat_dict_mapped, rank_to_charsB, len_intval)



cov_masks = (None, None)
print('Start:',datetime.datetime.today())
for i_charA in range(maxrange_to_tryA):
    #init
    mapped_scores[i_charA] = {}
    validchar_maskA = (mats[langA] == i_charA).any(-1)
    
    with Pool() as p:
        #init
        process_params = []
        cov_scores_dict = {}
        #Pour chaque caractère B transformant A
        for i_charB in range(maxrange_to_tryB):
            cov_scores = [0,0,{},None,None]
            valid_ranks = abs(i_charB-i_charA) < rank_limit
            n_already_mapped = sum([e == i_charB for e in rank_mapping.values()])
            valid_try = valid_ranks and n_already_mapped < mapping_limit_per_char_i
            valid_try = valid_try or current_rank_mapping[i_charA] == i_charB
            
            #check ranks compatibility
            if valid_try:
                
                #make the mapping
                #print(rank_to_charsA[i_charA], '=>', rank_to_charsB[i_charB],end=' | ')
                #print(i_charA, '=>', i_charB)
                current_rank_mapping = rank_mapping.copy()
                current_partialcov_mask = cov_masks[0].copy() if type(cov_masks[0]) != type(None) else None
                current_fullcov_mask = cov_masks[1].copy() if type(cov_masks[1]) != type(None) else None
                current_rank_mapping[i_charA] = i_charB
                
                #Crée les matrices mappées
                mapped_mat = create_mapped_mat(mats[langA], current_rank_mapping)
                validchar_maskB = (mats[langB] == i_charB).any(-1)
                
                #Calcule les scores
                params = mats[langB], mapped_mat, rank_to_charsB, len_intval, (validchar_maskA,validchar_maskB),\
                            (current_partialcov_mask,current_fullcov_mask)
                #process_params.append(params)
                cov_scores = p.apply_async(compute_coverages, params)
                #cov_scores = compute_coverages(*params)
                        
                #display scores
                #partial_word_coverage, full_word_coverage, found_words, _, _ = cov_scores
                #print('pc:',partial_word_coverage,'% | fc:',full_word_coverage, '%')
                #process_params.append(params)
            cov_scores_dict[i_charB] = (cov_scores,current_rank_mapping)
                
        
        for i_charB in range(maxrange_to_tryB):
            #enregistre les scores
            cov_scores = list(cov_scores_dict[i_charB][0].get()) if type(cov_scores_dict[i_charB][0]) != list else cov_scores_dict[i_charB][0]
            partial_word_coverage, full_word_coverage, found_words, _, _ = cov_scores
            
            if partial_word_coverage:
                print(rank_to_charsA[i_charA], '=>', rank_to_charsB[i_charB],end=' | ')
                print('pc:',partial_word_coverage,'% | fc:',full_word_coverage, '%')
            
            mapped_scores[i_charA][i_charB] = cov_scores
            mapped_scores[i_charA][i_charB].append(cov_scores_dict[i_charB][1])
        
    #Calcule et sélectionne le pic
    best_i_charB = np.argmax([mapped_scores[i_charA][c][0] for c in mapped_scores[i_charA].keys()])
    max_pc = np.max([mapped_scores[i_charA][c][0] for c in mapped_scores[i_charA].keys()])
    scores_nonzero = [mapped_scores[i_charA][c][0] for c in mapped_scores[i_charA].keys() if mapped_scores[i_charA][c][0] > 0]
    min_pc = np.min(scores_nonzero ) if len(scores_nonzero) > 0 else 0
    best_i_charB = -1 if min_pc == max_pc else best_i_charB
    
    #Display result
    if best_i_charB != -1:
        scores_i_charB = np.round(mapped_scores[i_charA][best_i_charB][0:2],2)
        
        rank_mapping[i_charA] = best_i_charB
        cov_masks = mapped_scores[i_charA][best_i_charB][3:5]
        
        #DISPLAY
        #print('Validating: ',i_charA,'=>',best_i_charB)
        print('Validating: ',rank_to_charsA[i_charA],'=>',rank_to_charsB[best_i_charB],\
              '(pc:',scores_i_charB[0],'% | fc:',scores_i_charB[1],'%)',sep=' ' )
        
        found_words = mapped_scores[i_charA][best_i_charB][2]
        found_words = [w for w,v in found_words.items() if v > 0.9]
        #print('_', len(found_words))
    else:
        print("Can't validate any decision")
        print('_')
    
    
    
    
    
#Appliquer le mapping
chars_mapping = OrderedDict({rank_to_charsA[rA]:rank_to_charsB[rB] \
                         for rA,rB in rank_mapping.items()})
#redo best to find mapping words
matching_words = foundWords(langA, langB, chars_mapping, **options)
transcripted = transcript(chars_mapping, langA)
dmatch = dict(matching_words)
print(' '.join([w+'*' if w in wordsA else w for w in transcripted[0:1000]]))
#print(loadWords(langA)[0:10])


