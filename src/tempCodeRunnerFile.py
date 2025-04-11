def acao_do_jogo(jogo_id):
    odds = apiclient.filtraOddsNovo([jogo_id])
    df_odds = apiclient.transform_betting_data(odds)
    df_odds = NN.preProcessGeneral(df_odds)