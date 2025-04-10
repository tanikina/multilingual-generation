#!/bin/sh
echo preparing_MASSIVE
python src/utils/preprocess_data_from_hf.py \
--dataset=massive \
--languages en-US de-DE th-TH he-IL id-ID sw-KE ro-RO az-AZ sl-SL te-IN cy-GB \
--intents alarm_query audio_volume_down calendar_remove cooking_recipe datetime_convert email_sendemail play_audiobook recommendation_movies transport_ticket weather_query
echo preparing_SIB-200
python src/utils/preprocess_data_from_hf.py \
--dataset=sib200 \
--languages eng_Latn deu_Latn tha_Thai heb_Hebr ind_Latn swh_Latn ron_Latn azj_Latn slv_Latn tel_Telu cym_Latn \
--intents science/technology travel politics sports health entertainment geography
echo preparing_Sentiment
python src/utils/preprocess_data_from_hf.py \
--dataset=sentiment \
--languages English_sentiment German_sentiment thai_sa hebrew_sa indonesian_sa swahili_sa romanian_sa azerbaijani_sa slovenian_sa telugu_sa welsh_sa \
--intents positive negative
