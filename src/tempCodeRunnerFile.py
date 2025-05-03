    dados_dataframe = dados_dataframe[dados_dataframe['horario'] > (agora - (7 * 60))]
        logger.info(f"📋 Jogos após filtragem por horário: {len(dados_dataframe)}")