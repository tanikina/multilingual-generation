yes_no_grammar = r"""
?start: action
action: operation done

done: " [e]"

operation: op1 |op2

op1: " yes"
op2: " no"
"""

intent_grammar_60 = r"""
?start: action
action: operation done

done: " [e]"

operation: op1 |op2 |op3 |op4 |op5 |op6 |op7 |op8 |op9 |op10 |op11 |op12 |op13 |op14 |op15 |op16 |op17 |op18 |op19 |op20 |op21 |op22 |op23 |op24 |op25 |op26 |op27 |op28 |op29 |op30 |op31 |op32 |op33 |op34 |op35 |op36 |op37 |op38 |op39 |op40 |op41 |op42 |op43 |op44 |op45 |op46 |op47 |op48 |op49 |op50 |op51 |op52 |op53 |op54 |op55 |op56 |op57 |op58 |op59

op1: " alarm_query"
op2: " alarm_remove"
op3: " alarm_set"
op4: " audio_volume_down"
op5: " audio_volume_mute"
op6: " audio_volume_up"
op7: " calendar_query"
op8: " calendar_remove"
op9: " calendar_set"
op10: " cooking_query"
op11: " cooking_recipe"
op12: " datetime_convert"
op13: " datetime_query"
op14: " email_addcontact"
op15: " email_query"
op16: " email_querycontact"
op17: " email_sendemail"
op18: " general_greet"
op19: " general_joke"
op20: " general_quirky"
op21: " iot_cleaning"
op22: " iot_coffee"
op23: " iot_hue_lightchange"
op24: " iot_hue_lightdim"
op25: " iot_hue_lightoff"
op26: " iot_hue_lighton"
op27: " iot_hue_lightup"
op28: " iot_wemo_off"
op29: " iot_wemo_on"
op30: " lists_createoradd"
op31: " lists_query"
op32: " lists_remove"
op33: " music_dislikeness"
op34: " music_likeness"
op35: " music_query"
op36: " music_settings"
op37: " news_query"
op38: " play_audiobook"
op39: " play_game"
op40: " play_music"
op41: " play_podcasts"
op42: " play_radio"
op43: " qa_currency"
op44: " qa_definition"
op45: " qa_factoid"
op46: " qa_maths"
op47: " qa_stock"
op48: " recommendation_events"
op49: " recommendation_locations"
op50: " recommendation_movies"
op51: " social_post"
op52: " social_query"
op53: " takeaway_order"
op54: " takeaway_query"
op55: " transport_query"
op56: " transport_taxi"
op57: " transport_ticket"
op58: " transport_traffic"
op59: " weather_query"
"""

intent_grammar_10 = r"""
?start: action
action: operation done

done: " [e]"

operation: op1 |op2 |op3 |op4 |op5 |op6 |op7 |op8 |op9 |op10

op1: " alarm_query"
op2: " audio_volume_down"
op3: " calendar_remove"
op4: " cooking_recipe"
op5: " datetime_convert"
op6: " email_sendemail"
op7: " play_audiobook"
op8: " recommendation_movies"
op9: " transport_ticket"
op10: " weather_query"
"""
