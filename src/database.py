import pandas as pd
import sqlite3
import csv
import os # Para verificar se o arquivo de banco de dados existe

# --- 1. Configurações do Banco de Dados ---
# O nome do arquivo do banco de dados SQLite. Ele será criado se não existir.
DB_FILE = "database.db" 

TABLE_NAME = "dados_resultados"  # Nome da tabela que será criada ou preenchida

CSV_FILE = "resultados_60.csv"

def create_table_from_csv_header(conn, file_path, table_name):
    """
    Cria uma tabela no banco de dados SQLite baseada nos cabeçalhos do CSV.
    Tenta inferir tipos de dados básicos (TEXT para tudo para simplificar).
    Para uma tipagem mais precisa, seria necessário analisar o conteúdo das colunas.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader) # Pega a primeira linha (cabeçalhos)

        # Remove caracteres especiais e espaços dos nomes das colunas para serem nomes de campos válidos
        columns = []
        for header in headers:
            # Substitui caracteres não alfanuméricos (exceto _) por '_' e garante que não comece com número se for o caso
            clean_header = "".join(c if c.isalnum() or c == '_' else '_' for c in header).lower()
            # Remove underscores duplicados e no início/fim, se houver
            clean_header = '_'.join(filter(None, clean_header.split('_')))
            if not clean_header: # Garante que o nome da coluna não fique vazio
                clean_header = f"coluna_{len(columns)}"
            # Adiciona um prefixo se começar com número (SQLite pode ter problemas)
            if clean_header and clean_header[0].isdigit():
                clean_header = "_" + clean_header
            columns.append(clean_header)

        # Cria a string de definição das colunas para a instrução CREATE TABLE
        # No SQLite, a maioria dos tipos de dados é armazenada como TEXT, INTEGER, REAL, BLOB, ou NULL.
        # Usar TEXT para tudo é um bom ponto de partida para CSVs.
        column_definitions = [f'"{col}" TEXT' for col in columns]
        
        create_table_query = f"CREATE TABLE IF NOT EXISTS \"{table_name}\" ({', '.join(column_definitions)})"
        
        conn.execute(create_table_query) 
        
        print(f"Tabela '{table_name}' verificada/criada com sucesso.")
        return columns # Retorna os nomes de colunas limpos para uso posterior
    except Exception as e:
        print(f"Erro ao criar/verificar tabela: {e}")
        return None

def insert_csv_to_db(file_path, table_name):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            print(f"Conexão com o banco de dados SQLite '{DB_FILE}' estabelecida com sucesso!")
            
            columns = create_table_from_csv_header(conn, file_path, table_name)
            if not columns:
                return

            df = pd.read_csv(file_path, encoding='utf-8')
            df.columns = [
                "".join(c if c.isalnum() or c == '_' else '_' for c in col).lower()
                for col in df.columns
            ]
            df.columns = ['_'.join(filter(None, col.split('_'))) for col in df.columns]
            df.columns = ['_' + col if col and col[0].isdigit() else col for col in df.columns]
            df = df[columns]

            placeholders = ', '.join(['?' for _ in columns])
            column_names_sql = ', '.join([f'"{col}"' for col in columns])
            insert_query = f"INSERT INTO \"{table_name}\" ({column_names_sql}) VALUES ({placeholders})"
            data_to_insert = [tuple(row) for row in df.itertuples(index=False)]

            print(f"Iniciando a inserção de {len(data_to_insert)} registros...")
            cursor = conn.cursor()
            try:
                cursor.executemany(insert_query, data_to_insert)
            finally:
                cursor.close()

            # --- Remove duplicatas pelo ID ---
            conn.execute(f"""
                DELETE FROM "{table_name}"
                WHERE ROWID NOT IN (
                    SELECT MIN(ROWID)
                    FROM "{table_name}"
                    GROUP BY id
                )
            """)
            print("Duplicatas removidas com sucesso.")

    except Exception as e:
        print(f"Erro: {e}")
    finally:
        print("Operação de banco de dados finalizada.")

# --- Execução do Script ---
if __name__ == "__main__":
    insert_csv_to_db(CSV_FILE, TABLE_NAME)