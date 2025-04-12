def pegaJogosDoDia():
    try:
        ids , tempo, time, times_id = apiclient.getUpcoming(leagues=apiclient.leagues_ids)
        
        if not ids:
            logger.warning("⚠️ Nenhum ID de jogo retornado pela API")
            return pd.DataFrame()
        # adicionar tratamento pra no caso de vazio
        dados = [{"id_jogo": i, "horario": h, "times": k, "home": z, "away": t} for i, h, k, z, t in zip(ids, tempo, time, times_id[0], times_id[1])]
        
        dados_dataframe = pd.DataFrame(dados)
        dados_dataframe = dados_dataframe[~dados_dataframe['id_jogo'].isin(programado)]
        if dados_dataframe.empty:
            logger.info("ℹ️ Todos os jogos já estão programados")
            return dados_dataframe
        
        dados_dataframe['horario'] = dados_dataframe['horario'].astype(int)
        dados_dataframe['send_time'] = dados_dataframe['horario'] - 350
        dados_dataframe = dados_dataframe.sort_values(by="horario").reset_index(drop=True)
        programados = dados_dataframe['id_jogo'].tolist()
        programado.extend(programados)
        logger.info(f"📌 Adicionados {len(programados)} novos jogos à lista de programados")
        dados_dataframe.to_csv('oque_sai_do_dadosDataframe.csv')
        return dados_dataframe
    
    except Exception as e:
        logger.error(f"❌ Erro ao obter jogos do dia: {str(e)}")
        return pd.DataFrame()
