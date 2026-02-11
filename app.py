import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import gradio as gr
from datetime import datetime, timedelta, time
import re
import warnings
import io
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ============================================
# CSS PERSONALIZADO - IDENTIDADE CHMSCS
# ============================================

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

.gradio-container {
    font-family: 'Inter', sans-serif;
    max-width: 1400px !important;
}

.header-hospital {
    background: linear-gradient(135deg, #6b4c9a 0%, #7c7adb 50%, #4dd4c4 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 2.5rem;
    box-shadow: 0 20px 60px rgba(107, 76, 154, 0.4);
}

.logos-container {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 2rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}

.logo-box {
    background: white;
    padding: 1rem 2rem;
    border-radius: 12px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.metric-card {
    background: white;
    padding: 2rem;
    border-radius: 16px;
    border-left: 5px solid #7c7adb;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin: 1.5rem 0;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 35px rgba(0,0,0,0.12);
    border-left-color: #4dd4c4;
}

.success-banner {
    background: linear-gradient(135deg, #4dd4c4 0%, #7c7adb 100%);
    padding: 2.5rem;
    border-radius: 16px;
    color: white;
    margin: 2rem 0;
    box-shadow: 0 15px 40px rgba(77, 212, 196, 0.3);
}

.warning-banner {
    background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%);
    padding: 2rem;
    border-radius: 16px;
    color: white;
    margin: 2rem 0;
    box-shadow: 0 15px 40px rgba(245, 158, 11, 0.3);
}

.stat-box {
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
    padding: 1.8rem;
    border-radius: 14px;
    border: 2px solid #7c7adb;
    text-align: center;
    transition: all 0.3s ease;
}

.stat-box:hover {
    border-color: #4dd4c4;
    transform: scale(1.05);
}

.stat-number {
    font-size: 2.8rem;
    font-weight: 800;
    color: #6b4c9a;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
    margin: 2.5rem 0;
}

.feature-item {
    background: white;
    padding: 2rem;
    border-radius: 14px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    text-align: center;
    border-top: 4px solid #7c7adb;
    transition: all 0.3s ease;
}

.feature-item:hover {
    border-top-color: #4dd4c4;
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}
"""

# ============================================
# FUNÇÕES BASE
# ============================================

def parse_time(time_value):
    if pd.isna(time_value):
        return None
    if isinstance(time_value, time):
        return time_value
    if isinstance(time_value, datetime):
        return time_value.time()
    time_str = str(time_value).strip()
    try:
        if ':' in time_str:
            dt = pd.to_datetime(time_str, format='%H:%M:%S', errors='coerce')
            if pd.isna(dt):
                dt = pd.to_datetime(time_str, format='%H:%M', errors='coerce')
            if pd.notna(dt):
                return dt.time()
        dt = pd.to_datetime(time_str, errors='coerce')
        if pd.notna(dt):
            return dt.time()
    except:
        pass
    return None

def converter_duracao_para_minutos(duracao_str):
    if pd.isna(duracao_str) or duracao_str == '':
        return 0
    duracao_str = str(duracao_str).upper().strip()
    
    if 'HORA' in duracao_str:
        match = re.search(r'(\d+)', duracao_str)
        if match:
            return int(match.group(1)) * 60
    
    if 'MINUTO' in duracao_str and 'HORA' not in duracao_str:
        match = re.search(r'(\d+)', duracao_str)
        if match:
            return int(match.group(1))
    
    match = re.search(r'(\d+):(\d+)', duracao_str)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    
    try:
        return int(float(duracao_str))
    except:
        return 0

def processar_grade_cirurgica(file_path):
    todos_dados = []
    xl = pd.ExcelFile(file_path)
    
    for sheet_name in xl.sheet_names:
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        dia_semana = sheet_name.strip().upper().replace('_', ' ').title()
        sala_atual = None
        
        for idx, row in df_raw.iterrows():
            primeira_celula = str(row[0]).strip().upper() if pd.notna(row[0]) else ''
            
            if 'SALA' in primeira_celula and 'CIRÚRGICA' in primeira_celula:
                sala_atual = primeira_celula.replace('SALA CIRÚRGICA', 'SALA').strip()
                continue
            
            if 'HORÁRIO' in primeira_celula or 'HÓARIO' in primeira_celula:
                if sala_atual:
                    for data_idx in range(idx + 1, min(idx + 20, len(df_raw))):
                        data_row = df_raw.iloc[data_idx]
                        horario = data_row[0] if len(data_row) > 0 else None
                        
                        if pd.isna(horario) or 'URGÊNCIA' in str(horario).upper():
                            break
                        
                        especialidade = data_row[1] if len(data_row) > 1 else 'GERAL'
                        duracao = data_row[2] if len(data_row) > 2 else None
                        
                        if pd.notna(horario) and str(horario).strip() != '':
                            todos_dados.append({
                                'dia': dia_semana,
                                'sala': sala_atual,
                                'horario_inicio': horario,
                                'especialidade': str(especialidade) if pd.notna(especialidade) else 'GERAL',
                                'duracao_minutos': converter_duracao_para_minutos(duracao)
                            })
    
    df_final = pd.DataFrame(todos_dados)
    if len(df_final) == 0:
        return None
    
    df_final['horario_inicio_time'] = df_final['horario_inicio'].apply(parse_time)
    
    def calcular_fim(row):
        if pd.notna(row['horario_inicio_time']) and row['duracao_minutos'] > 0:
            inicio_dt = datetime.combine(datetime.today(), row['horario_inicio_time'])
            fim_dt = inicio_dt + timedelta(minutes=row['duracao_minutos'])
            return fim_dt.time()
        return None
    
    df_final['horario_fim'] = df_final.apply(calcular_fim, axis=1)
    return df_final

# ============================================
# ANÁLISE UNIFICADA
# ============================================

def analisar_grade_e_gaps(file_grade):
    if file_grade is None:
        return "❌ Faça upload da grade", None, None, None, None, None
    
    try:
        df = processar_grade_cirurgica(file_grade.name)
        
        if df is None:
            return "❌ Erro ao processar", None, None, None, None, None
        
        total = len(df)
        especialidades = df['especialidade'].nunique()
        salas = df['sala'].nunique()
        dias = df['dia'].nunique()
        
        ocupacao_sala = df.groupby('sala')['duracao_minutos'].sum()
        ocupacao_pct = (ocupacao_sala / (600 * dias) * 100).round(1)
        ocupacao_media = ocupacao_pct.mean()
        
        dist_esp = df.groupby('especialidade').agg({
            'duracao_minutos': ['sum', 'mean', 'count']
        }).round(1)
        dist_esp.columns = ['Total (min)', 'Média (min)', 'Quantidade']
        dist_esp = dist_esp.reset_index().sort_values('Quantidade', ascending=False)
        
        # GAPS
        gaps = []
        
        for (dia, sala), grupo in df.groupby(['dia', 'sala']):
            grupo = grupo.dropna(subset=['horario_inicio_time']).sort_values('horario_inicio_time')
            
            if len(grupo) > 0:
                primeira = grupo.iloc[0]
                inicio_exp = time(8, 0)
                
                if primeira['horario_inicio_time'] > inicio_exp:
                    inicio_dt = datetime.combine(datetime.today(), inicio_exp)
                    primeira_dt = datetime.combine(datetime.today(), primeira['horario_inicio_time'])
                    gap_min = (primeira_dt - inicio_dt).total_seconds() / 60
                    
                    if gap_min >= 30:
                        gaps.append({
                            'Dia': dia, 'Sala': sala,
                            'Início': str(inicio_exp), 'Fim': str(primeira['horario_inicio_time']),
                            'Duração (min)': int(gap_min), 'Tipo': '🌅 INÍCIO'
                        })
                
                for i in range(len(grupo) - 1):
                    fim = grupo.iloc[i]['horario_fim']
                    inicio_prox = grupo.iloc[i + 1]['horario_inicio_time']
                    
                    if pd.notna(fim) and pd.notna(inicio_prox):
                        try:
                            fim_dt = datetime.combine(datetime.today(), fim)
                            inicio_dt = datetime.combine(datetime.today(), inicio_prox)
                            gap_min = (inicio_dt - fim_dt).total_seconds() / 60
                            
                            if gap_min >= 30:
                                gaps.append({
                                    'Dia': dia, 'Sala': sala,
                                    'Início': str(fim), 'Fim': str(inicio_prox),
                                    'Duração (min)': int(gap_min), 'Tipo': '⏸️ ENTRE'
                                })
                        except:
                            pass
                
                ultima = grupo.iloc[-1]
                fim_exp = time(18, 0)
                
                if pd.notna(ultima['horario_fim']) and ultima['horario_fim'] &lt; fim_exp:
                    ultima_dt = datetime.combine(datetime.today(), ultima['horario_fim'])
                    fim_dt = datetime.combine(datetime.today(), fim_exp)
                    gap_min = (fim_dt - ultima_dt).total_seconds() / 60
                    
                    if gap_min >= 30:
                        gaps.append({
                            'Dia': dia, 'Sala': sala,
                            'Início': str(ultima['horario_fim']), 'Fim': str(fim_exp),
                            'Duração (min)': int(gap_min), 'Tipo': '🌆 FIM'
                        })
        
        gaps_df = pd.DataFrame(gaps)
        total_gap_min = gaps_df['Duração (min)'].sum() if len(gaps_df) > 0 else 0
        
        # Gráficos
        fig1 = px.bar(
            dist_esp, x='Quantidade', y='especialidade',
            title='📊 Cirurgias por Especialidade',
            orientation='h', text='Quantidade',
            color='Total (min)', 
            color_continuous_scale=[[0, '#4dd4c4'], [0.5, '#7c7adb'], [1, '#6b4c9a']]
        )
        fig1.update_traces(textposition='outside')
        fig1.update_layout(height=500, template='plotly_white')
        
        ocupacao_df = pd.DataFrame({'Sala': ocupacao_pct.index, 'Ocupação (%)': ocupacao_pct.values})
        
        fig2 = px.bar(
            ocupacao_df, x='Sala', y='Ocupação (%)',
            title='📈 Ocupação por Sala',
            text='Ocupação (%)', color='Ocupação (%)',
            color_continuous_scale=[[0, '#fecaca'], [0.5, '#fcd34d'], [1, '#4dd4c4']]
        )
        fig2.add_hline(y=70, line_dash="dash", line_color="#6b4c9a", annotation_text="Meta: 70%")
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2.update_layout(height=480, template='plotly_white')
        
        if len(gaps_df) > 0:
            gaps_tipo = gaps_df.groupby('Tipo')['Duração (min)'].sum().reset_index()
            fig3 = px.pie(gaps_tipo, values='Duração (min)', names='Tipo',
                         title='🕳️ Gaps por Tipo',
                         color_discrete_sequence=['#4dd4c4', '#7c7adb', '#6b4c9a'])
            fig3.update_traces(textposition='inside', textinfo='percent+label')
            fig3.update_layout(height=450)
        else:
            fig3 = None
        
        resumo = f"""
<div class="success-banner">
    <h2 style="margin: 0;">✅ Análise Completa!</h2>
    <p style="margin-top: 1rem; font-size: 1.2rem;">Grade + Gaps analisados</p>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1.5rem; margin: 2rem 0;">
    <div class="stat-box"><p class="stat-number">{total}</p><p class="stat-label">Cirurgias/Semana</p></div>
    <div class="stat-box"><p class="stat-number">{salas}</p><p class="stat-label">Salas</p></div>
    <div class="stat-box"><p class="stat-number">{ocupacao_media:.1f}%</p><p class="stat-label">Ocupação</p></div>
    <div class="stat-box"><p class="stat-number">{len(gaps_df)}</p><p class="stat-label">Gaps</p></div>
    <div class="stat-box"><p class="stat-number">{total_gap_min}</p><p class="stat-label">Min Disponíveis</p></div>
</div>
        """
        
        return resumo, dist_esp, gaps_df, fig1, fig2, fig3
        
    except Exception as e:
        return f"❌ Erro: {str(e)}", None, None, None, None, None

# ============================================
# SISTEMA DE SLOTS
# ============================================

def criar_grade_com_slots(file_grade):
    if file_grade is None:
        return "❌ Upload necessário", None, None, None
    
    try:
        df = processar_grade_cirurgica(file_grade.name)
        if df is None:
            return "❌ Erro", None, None, None
        
        slots_totais = 10
        grade_slots = []
        
        for (dia, sala), grupo in df.groupby(['dia', 'sala']):
            grupo = grupo.dropna(subset=['horario_inicio_time']).sort_values('horario_inicio_time')
            slots = ['🟢 LIVRE'] * slots_totais
            
            for _, cirurgia in grupo.iterrows():
                if pd.notna(cirurgia['horario_inicio_time']):
                    hora_inicio = cirurgia['horario_inicio_time'].hour
                    minuto_inicio = cirurgia['horario_inicio_time'].minute
                    slot_inicio = hora_inicio - 8 + (minuto_inicio / 60)
                    num_slots = int(np.ceil(cirurgia['duracao_minutos'] / 60))
                    
                    slot_idx = int(slot_inicio)
                    for i in range(num_slots):
                        if 0 <= slot_idx + i < slots_totais:
                            esp_curta = cirurgia['especialidade'][:12]
                            slots[slot_idx + i] = f"🔴 {esp_curta}"
            
            grade_slots.append({
                'Dia': dia, 'Sala': sala,
                '08h': slots[0], '09h': slots[1], '10h': slots[2], '11h': slots[3], '12h': slots[4],
                '13h': slots[5], '14h': slots[6], '15h': slots[7], '16h': slots[8], '17h': slots[9]
            })
        
        df_slots = pd.DataFrame(grade_slots)
        
        total_slots = len(df_slots) * slots_totais
        slots_livres = sum([1 for _, row in df_slots.iterrows() 
                           for hora in ['08h', '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h']
                           if '🟢' in str(row[hora])])
        
        slots_pequeno = int(slots_livres * 1)
        slots_medio = int(slots_livres / 2)
        slots_grande = int(slots_livres / 3)
        
        disp_slots = pd.DataFrame({
            'Porte': ['Pequeno (60-90min)', 'Médio (120-180min)', 'Grande (240-300min)'],
            'Slots Necessários': [1, 2, 3],
            'Cirurgias Possíveis': [slots_pequeno, slots_medio, slots_grande]
        })
        
        fig = px.bar(disp_slots, x='Porte', y='Cirurgias Possíveis',
                     title='🎯 Capacidade por Porte', text='Cirurgias Possíveis',
                     color='Cirurgias Possíveis',
                     color_continuous_scale=[[0, '#4dd4c4'], [1, '#6b4c9a']])
        fig.update_traces(textposition='outside')
        fig.update_layout(height=450, template='plotly_white')
        
        resumo = f"""
<div class="success-banner">
    <h2 style="margin: 0;">✅ Grade de Slots Criada!</h2>
</div>

<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin: 2rem 0;">
    <div class="stat-box"><p class="stat-number">{total_slots}</p><p class="stat-label">Slots Totais</p></div>
    <div class="stat-box"><p class="stat-number" style="color: #22c55e;">{slots_livres}</p><p class="stat-label">Livres</p></div>
    <div class="stat-box"><p class="stat-number">{(total_slots-slots_livres)/total_slots*100:.1f}%</p><p class="stat-label">Ocupação</p></div>
    <div class="stat-box"><p class="stat-number">~{slots_medio}</p><p class="stat-label">Cirurgias Médias</p></div>
</div>
        """
        
        return resumo, df_slots, disp_slots, fig
        
    except Exception as e:
        return f"❌ Erro: {str(e)}", None, None, None

# ============================================
# A) ALOCAÇÃO AUTOMÁTICA POR SLOTS
# ============================================

def processar_fila_cirurgias(file_fila):
    """Processa arquivo de fila de cirurgias"""
    if file_fila is None:
        return None
    
    df = pd.read_excel(file_fila.name)
    
    fila = []
    for idx, row in df.iterrows():
        paciente = str(row.get('Paciente', f'Paciente_{idx+1}'))
        esp = str(row.get('Especialidade', 'GERAL'))
        dur = int(row.get('Duracao', row.get('Duração', 120)))
        
        # Calcular slots necessários
        slots_necessarios = int(np.ceil(dur / 60))
        
        # Classificar porte
        if dur < 120:
            porte = 'Pequeno'
        elif dur < 240:
            porte = 'Médio'
        else:
            porte = 'Grande'
        
        fila.append({
            'Paciente': paciente,
            'Especialidade': esp,
            'Duração (min)': dur,
            'Porte': porte,
            'Slots Necessários': slots_necessarios
        })
    
    return pd.DataFrame(fila)

def alocar_automaticamente_por_slots(file_grade, file_fila):
    """A) ALOCAÇÃO AUTOMÁTICA - Aloca cirurgias da fila nos slots disponíveis"""
    
    if file_grade is None or file_fila is None:
        return "❌ Faça upload de ambos os arquivos", None, None
    
    try:
        df_grade = processar_grade_cirurgica(file_grade.name)
        df_fila = processar_fila_cirurgias(file_fila)
        
        if df_grade is None or df_fila is None:
            return "❌ Erro ao processar", None, None
        
        # Identificar gaps
        gaps = []
        for (dia, sala), grupo in df_grade.groupby(['dia', 'sala']):
            grupo = grupo.dropna(subset=['horario_inicio_time']).sort_values('horario_inicio_time')
            
            if len(grupo) > 0:
                primeira = grupo.iloc[0]
                inicio_exp = time(8, 0)
                
                if primeira['horario_inicio_time'] > inicio_exp:
                    inicio_dt = datetime.combine(datetime.today(), inicio_exp)
                    primeira_dt = datetime.combine(datetime.today(), primeira['horario_inicio_time'])
                    gap_min = (primeira_dt - inicio_dt).total_seconds() / 60
                    
                    if gap_min >= 60:
                        gaps.append({
                            'dia': dia, 'sala': sala,
                            'horario_inicio': inicio_exp,
                            'duracao_gap': int(gap_min)
                        })
                
                for i in range(len(grupo) - 1):
                    fim = grupo.iloc[i]['horario_fim']
                    inicio_prox = grupo.iloc[i + 1]['horario_inicio_time']
                    
                    if pd.notna(fim) and pd.notna(inicio_prox):
                        try:
                            fim_dt = datetime.combine(datetime.today(), fim)
                            inicio_dt = datetime.combine(datetime.today(), inicio_prox)
                            gap_min = (inicio_dt - fim_dt).total_seconds() / 60
                            
                            if gap_min >= 60:
                                gaps.append({
                                    'dia': dia, 'sala': sala,
                                    'horario_inicio': fim,
                                    'duracao_gap': int(gap_min)
                                })
                        except:
                            pass
        
        gaps_df = pd.DataFrame(gaps)
        
        # Alocar cirurgias
        alocacoes = []
        
        for idx, cirurgia in df_fila.iterrows():
            duracao_necessaria = cirurgia['Duração (min)']
            
            gaps_compativeis = gaps_df[gaps_df['duracao_gap'] >= duracao_necessaria].copy()
            
            if len(gaps_compativeis) == 0:
                continue
            
            gaps_compativeis['desperdicio'] = gaps_compativeis['duracao_gap'] - duracao_necessaria
            melhor_gap = gaps_compativeis.sort_values('desperdicio').iloc[0]
            
            alocacoes.append({
                'Paciente': cirurgia['Paciente'],
                'Especialidade': cirurgia['Especialidade'],
                'Porte': cirurgia['Porte'],
                'Duração': cirurgia['Duração (min)'],
                'Slots': cirurgia['Slots Necessários'],
                'Dia': melhor_gap['dia'],
                'Sala': melhor_gap['sala'],
                'Horário': str(melhor_gap['horario_inicio']),
                'Gap Original': melhor_gap['duracao_gap']
            })
            
            gap_idx = gaps_df[
                (gaps_df['dia'] == melhor_gap['dia']) & 
                (gaps_df['sala'] == melhor_gap['sala']) &
                (gaps_df['horario_inicio'] == melhor_gap['horario_inicio'])
            ].index
            gaps_df = gaps_df.drop(gap_idx)
            
            tempo_restante = melhor_gap['duracao_gap'] - duracao_necessaria
            if tempo_restante >= 60:
                if isinstance(melhor_gap['horario_inicio'], time):
                    novo_inicio_dt = datetime.combine(datetime.today(), melhor_gap['horario_inicio'])
                else:
                    novo_inicio_dt = datetime.strptime(str(melhor_gap['horario_inicio']), '%H:%M:%S')
                
                novo_inicio = (novo_inicio_dt + timedelta(minutes=duracao_necessaria)).time()
                
                novo_gap = pd.DataFrame([{
                    'dia': melhor_gap['dia'],
                    'sala': melhor_gap['sala'],
                    'horario_inicio': novo_inicio,
                    'duracao_gap': int(tempo_restante)
                }])
                gaps_df = pd.concat([gaps_df, novo_gap], ignore_index=True)
        
        alocacoes_df = pd.DataFrame(alocacoes)
        
        resumo = f"""
<div class="success-banner">
    <h2 style="margin: 0;">✅ Alocação Automática Concluída!</h2>
    <p style="margin-top: 1rem; font-size: 1.2rem;">{len(alocacoes_df)} cirurgias alocadas automaticamente</p>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1.5rem; margin: 2rem 0;">
    <div class="stat-box"><p class="stat-number">{len(df_fila)}</p><p class="stat-label">Fila Original</p></div>
    <div class="stat-box"><p class="stat-number" style="color: #22c55e;">{len(alocacoes_df)}</p><p class="stat-label">Alocadas</p></div>
    <div class="stat-box"><p class="stat-number">{len(df_fila)-len(alocacoes_df)}</p><p class="stat-label">Pendentes</p></div>
    <div class="stat-box"><p class="stat-number">{len(alocacoes_df)/len(df_fila)*100:.1f}%</p><p class="stat-label">Taxa Alocação</p></div>
</div>
        """
        
        return resumo, alocacoes_df
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"❌ Erro: {str(e)}", None

# ============================================
# B) VISUALIZAÇÃO TIMELINE (GANTT)
# ============================================

def criar_timeline_visual(file_grade):
    """B) TIMELINE - Visualização estilo Gantt da grade"""
    
    if file_grade is None:
        return "❌ Upload necessário", None
    
    try:
        df = processar_grade_cirurgica(file_grade.name)
        if df is None:
            return "❌ Erro", None
        
        # Preparar dados para Gantt
        df_gantt = df.dropna(subset=['horario_inicio_time', 'horario_fim']).copy()
        
        timeline_data = []
        
        for idx, row in df_gantt.iterrows():
            # Criar datetime fictício para plotagem
            data_base = datetime(2026, 1, 1)
            
            inicio_dt = datetime.combine(data_base, row['horario_inicio_time'])
            fim_dt = datetime.combine(data_base, row['horario_fim'])
            
            timeline_data.append({
                'Task': f"{row['sala']} - {row['dia']}",
                'Start': inicio_dt,
                'Finish': fim_dt,
                'Resource': row['especialidade']
            })
        
        df_timeline = pd.DataFrame(timeline_data)
        
        # Criar Gantt
        fig = px.timeline(
            df_timeline, x_start='Start', x_end='Finish', y='Task', color='Resource',
            title='📅 Timeline Visual da Grade Cirúrgica - CHMSCS',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_yaxes(categoryorder='category ascending')
        fig.update_layout(
            height=800,
            xaxis_title='Horário',
            yaxis_title='Sala / Dia',
            template='plotly_white',
            font=dict(size=11)
        )
        
        resumo = f"""
<div class="success-banner">
    <h2 style="margin: 0;">✅ Timeline Criada!</h2>
    <p style="margin-top: 1rem;">Visualização Gantt de {len(df_gantt)} cirurgias</p>
</div>

<div class="metric-card">
    <h3 style="color: #6b4c9a; margin-top: 0;">💡 Como Interpretar</h3>
    <p style="font-size: 1.1rem;">
        Cada barra representa uma cirurgia. A cor indica a especialidade.
        Espaços vazios são gaps que podem ser preenchidos!
    </p>
</div>
        """
        
        return resumo, fig
        
    except Exception as e:
        return f"❌ Erro: {str(e)}", None

# ============================================
# C) SIMULADOR DE ALOCAÇÃO
# ============================================

def simular_alocacao(file_grade, num_pequeno, num_medio, num_grande):
    """C) SIMULADOR - Testa alocação de N cirurgias"""
    
    if file_grade is None:
        return "❌ Upload necessário", None
    
    try:
        df = processar_grade_cirurgica(file_grade.name)
        if df is None:
            return "❌ Erro", None
        
        # Calcular slots livres
        slots_totais = 10
        grade_slots = []
        
        for (dia, sala), grupo in df.groupby(['dia', 'sala']):
            grupo = grupo.dropna(subset=['horario_inicio_time']).sort_values('horario_inicio_time')
            slots = [True] * slots_totais  # True = livre
            
            for _, cirurgia in grupo.iterrows():
                if pd.notna(cirurgia['horario_inicio_time']):
                    hora_inicio = cirurgia['horario_inicio_time'].hour
                    slot_idx = hora_inicio - 8
                    num_slots = int(np.ceil(cirurgia['duracao_minutos'] / 60))
                    
                    for i in range(num_slots):
                        if 0 <= slot_idx + i < slots_totais:
                            slots[slot_idx + i] = False  # Ocupado
            
            grade_slots.extend(slots)
        
        slots_livres = sum(grade_slots)
        
        # Simular alocação
        slots_pequeno_necessarios = num_pequeno * 1
        slots_medio_necessarios = num_medio * 2
        slots_grande_necessarios = num_grande * 3
        
        slots_totais_necessarios = slots_pequeno_necessarios + slots_medio_necessarios + slots_grande_necessarios
        
        slots_restantes = slots_livres - slots_totais_necessarios
        
        viavel = slots_restantes >= 0
        
        if viavel:
            banner_class = "success-banner"
            emoji = "✅"
            mensagem = "Alocação VIÁVEL!"
        else:
            banner_class = "warning-banner"
            emoji = "⚠️"
            mensagem = "Alocação NÃO VIÁVEL - Faltam slots!"
        
        resumo = f"""
<div class="{banner_class}">
    <h2 style="margin: 0;">{emoji} {mensagem}</h2>
    <p style="margin-top: 1rem; font-size: 1.2rem;">Simulação de alocação concluída</p>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin: 2rem 0;">
    
    <div class="stat-box">
        <p class="stat-number">{slots_livres}</p>
        <p class="stat-label">Slots Disponíveis</p>
    </div>
    
    <div class="stat-box">
        <p class="stat-number">{slots_totais_necessarios}</p>
        <p class="stat-label">Slots Necessários</p>
    </div>
    
    <div class="stat-box">
        <p class="stat-number" style="color: {'#22c55e' if viavel else '#dc2626'};">{slots_restantes}</p>
        <p class="stat-label">Slots Restantes</p>
    </div>
    
    <div class="stat-box">
        <p class="stat-number">{num_pequeno + num_medio + num_grande}</p>
        <p class="stat-label">Total Cirurgias</p>
    </div>

</div>

<div class="metric-card">
    <h3 style="color: #6b4c9a; margin-top: 0;">📊 Detalhamento da Simulação</h3>
    <table style="width: 100%; border-collapse: collapse;">
        <thead>
            <tr style="background: #f1f5f9;">
                <th style="padding: 0.8rem; text-align: left;">Porte</th>
                <th style="padding: 0.8rem; text-align: center;">Quantidade</th>
                <th style="padding: 0.8rem; text-align: center;">Slots/Unidade</th>
                <th style="padding: 0.8rem; text-align: center;">Slots Totais</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 0.6rem;">🟢 Pequeno</td>
                <td style="padding: 0.6rem; text-align: center;">{num_pequeno}</td>
                <td style="padding: 0.6rem; text-align: center;">1</td>
                <td style="padding: 0.6rem; text-align: center; font-weight: 600;">{slots_pequeno_necessarios}</td>
            </tr>
            <tr>
                <td style="padding: 0.6rem;">🟡 Médio</td>
                <td style="padding: 0.6rem; text-align: center;">{num_medio}</td>
                <td style="padding: 0.6rem; text-align: center;">2</td>
                <td style="padding: 0.6rem; text-align: center; font-weight: 600;">{slots_medio_necessarios}</td>
            </tr>
            <tr>
                <td style="padding: 0.6rem;">🔴 Grande</td>
                <td style="padding: 0.6rem; text-align: center;">{num_grande}</td>
                <td style="padding: 0.6rem; text-align: center;">3</td>
                <td style="padding: 0.6rem; text-align: center; font-weight: 600;">{slots_grande_necessarios}</td>
            </tr>
            <tr style="background: #f8fafc; font-weight: 700;">
                <td style="padding: 0.8rem;">TOTAL</td>
                <td style="padding: 0.8rem; text-align: center;">{num_pequeno + num_medio + num_grande}</td>
                <td style="padding: 0.8rem; text-align: center;">-</td>
                <td style="padding: 0.8rem; text-align: center; color: #6b4c9a;">{slots_totais_necessarios}</td>
            </tr>
        </tbody>
    </table>
</div>

<div class="metric-card">
    <h3 style="color: #6b4c9a; margin-top: 0;">💡 Resultado</h3>
    <p style="font-size: 1.15rem; line-height: 1.8;">
        {'✅ <strong>VIÁVEL!</strong> Após alocar todas as cirurgias, sobrarão ' + str(slots_restantes) + ' slots livres.' if viavel else '❌ <strong>NÃO VIÁVEL!</strong> Faltam ' + str(abs(slots_restantes)) + ' slots. Reduza o número de cirurgias ou aumente a capacidade.'}
    </p>
</div>
        """
        
        # Criar tabela de simulação
        simulacao_df = pd.DataFrame({
            'Porte': ['Pequeno', 'Médio', 'Grande', 'TOTAL'],
            'Quantidade': [num_pequeno, num_medio, num_grande, num_pequeno + num_medio + num_grande],
            'Slots/Unidade': [1, 2, 3, '-'],
            'Slots Totais': [slots_pequeno_necessarios, slots_medio_necessarios, 
                            slots_grande_necessarios, slots_totais_necessarios]
        })
        
        return resumo, simulacao_df
        
    except Exception as e:
        return f"❌ Erro: {str(e)}", None

# ============================================
# D) EXPORTAR GRADE OTIMIZADA
# ============================================

def exportar_grade_otimizada(file_grade, file_fila):
    """D) EXPORTAR - Gera Excel com grade otimizada"""
    
    if file_grade is None:
        return "❌ Upload da grade necessário", None
    
    try:
        df_grade = processar_grade_cirurgica(file_grade.name)
        if df_grade is None:
            return "❌ Erro ao processar grade", None
        
        # Se tiver fila, alocar automaticamente
        if file_fila is not None:
            df_fila = processar_fila_cirurgias(file_fila)
            
            # Alocar (código simplificado)
            gaps = []
            for (dia, sala), grupo in df_grade.groupby(['dia', 'sala']):
                grupo = grupo.dropna(subset=['horario_inicio_time']).sort_values('horario_inicio_time')
                
                if len(grupo) > 0:
                    for i in range(len(grupo) - 1):
                        fim = grupo.iloc[i]['horario_fim']
                        inicio_prox = grupo.iloc[i + 1]['horario_inicio_time']
                        
                        if pd.notna(fim) and pd.notna(inicio_prox):
                            try:
                                fim_dt = datetime.combine(datetime.today(), fim)
                                inicio_dt = datetime.combine(datetime.today(), inicio_prox)
                                gap_min = (inicio_dt - fim_dt).total_seconds() / 60
                                
                                if gap_min >= 60:
                                    gaps.append({
                                        'dia': dia, 'sala': sala,
                                        'horario_inicio': fim,
                                        'duracao_gap': int(gap_min)
                                    })
                            except:
                                pass
            
            gaps_df = pd.DataFrame(gaps)
            
            # Alocar cirurgias
            for idx, cirurgia in df_fila.iterrows():
                duracao = cirurgia['Duração (min)']
                
                gaps_compativeis = gaps_df[gaps_df['duracao_gap'] >= duracao].copy()
                
                if len(gaps_compativeis) > 0:
                    melhor = gaps_compativeis.iloc[0]
                    
                    # Adicionar à grade
                    nova_linha = {
                        'dia': melhor['dia'],
                        'sala': melhor['sala'],
                        'horario_inicio': str(melhor['horario_inicio']),
                        'especialidade': f"NOVA: {cirurgia['Especialidade']}",
                        'duracao_minutos': duracao,
                        'horario_inicio_time': melhor['horario_inicio'] if isinstance(melhor['horario_inicio'], time) else parse_time(melhor['horario_inicio'])
                    }
                    
                    df_grade = pd.concat([df_grade, pd.DataFrame([nova_linha])], ignore_index=True)
                    
                    # Remover gap usado
                    gaps_df = gaps_df.drop(gaps_df.index[0])
        
        # Organizar e formatar para exportação
        df_export = df_grade.sort_values(['dia', 'sala', 'horario_inicio_time']).copy()
        
        df_export['Horário Início'] = df_export['horario_inicio'].astype(str)
        df_export['Horário Fim'] = df_export['horario_fim'].apply(lambda x: str(x) if pd.notna(x) else '')
        df_export['Duração (min)'] = df_export['duracao_minutos']
        df_export['Especialidade'] = df_export['especialidade']
        df_export['Dia'] = df_export['dia']
        df_export['Sala'] = df_export['sala']
        
        df_final = df_export[['Dia', 'Sala', 'Horário Início', 'Horário Fim', 
                              'Duração (min)', 'Especialidade']]
        
        # Criar Excel em memória
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_final.to_excel(writer, sheet_name='Grade Otimizada', index=False)
        
        output.seek(0)
        
        resumo = f"""
<div class="success-banner">
    <h2 style="margin: 0;">✅ Grade Otimizada Gerada!</h2>
    <p style="margin-top: 1rem; font-size: 1.2rem;">Arquivo Excel pronto para download</p>
</div>

<div class="metric-card">
    <h3 style="color: #6b4c9a; margin-top: 0;">📊 Conteúdo do Arquivo</h3>
    <ul style="font-size: 1.1rem; line-height: 1.8;">
        <li><strong>Total de cirurgias:</strong> {len(df_final)}</li>
        <li><strong>Formato:</strong> Excel (.xlsx)</li>
        <li><strong>Colunas:</strong> Dia, Sala, Horário Início, Horário Fim, Duração, Especialidade</li>
        <li><strong>Ordenação:</strong> Por dia, sala e horário</li>
    </ul>
    <p style="font-size: 1.05rem; margin-top: 1rem; color: #64748b;">
        ⬇️ Use o botão de download abaixo para salvar o arquivo
    </p>
</div>
        """
        
        return resumo, output
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"❌ Erro: {str(e)}", None

# ============================================
# ML
# ============================================

def extrair_dados_ml_da_grade(df_grade):
    if df_grade is None or len(df_grade) == 0:
        return None
    
    dados_ml = []
    
    for idx, row in df_grade.iterrows():
        if row['duracao_minutos'] > 0:
            duracao = row['duracao_minutos']
            
            if duracao < 120:
                porte, base = 'I', 90
            elif duracao < 240:
                porte, base = 'II', 180
            elif duracao < 360:
                porte, base = 'III', 300
            else:
                porte, base = 'IV', 420
            
            turno = 'MANHÃ' if (pd.notna(row['horario_inicio_time']) and row['horario_inicio_time'].hour < 13) else 'TARDE'
            
            dados_ml.append({
                'Especialidade': row['especialidade'],
                'Porte': porte,
                'Turno': turno,
                'Duracao_Planejada': base,
                'Duracao_Real': row['duracao_minutos']
            })
    
    return pd.DataFrame(dados_ml)

def treinar_modelo_chmscs(file_grade):
    if file_grade is None:
        return "❌ Upload necessário", None, None, None, None, None, None
    
    try:
        df_grade = processar_grade_cirurgica(file_grade.name)
        
        if df_grade is None or len(df_grade) < 20:
            return "❌ Poucos dados", None, None, None, None, None, None
        
        df_ml = extrair_dados_ml_da_grade(df_grade)
        
        le_esp = LabelEncoder()
        le_porte = LabelEncoder()
        le_turno = LabelEncoder()
        
        df_ml['esp_enc'] = le_esp.fit_transform(df_ml['Especialidade'])
        df_ml['porte_enc'] = le_porte.fit_transform(df_ml['Porte'])
        df_ml['turno_enc'] = le_turno.fit_transform(df_ml['Turno'])
        
        X = df_ml[['esp_enc', 'porte_enc', 'turno_enc', 'Duracao_Planejada']]
        y = df_ml['Duracao_Real']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        modelo = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        modelo.fit(X_train, y_train)
        
        y_pred = modelo.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        importancias = pd.DataFrame({
            'Variável': ['Especialidade', 'Porte', 'Turno', 'Duração Planejada'],
            'Importância (%)': (modelo.feature_importances_ * 100).round(1)
        }).sort_values('Importância (%)', ascending=False)
        
        analise_esp = df_ml.groupby('Especialidade').agg({
            'Duracao_Real': ['mean', 'std', 'count']
        }).round(1)
        analise_esp.columns = ['Média (min)', 'Desvio Padrão', 'Quantidade']
        analise_esp = analise_esp.reset_index().sort_values('Quantidade', ascending=False)
        
        fig1 = px.scatter(x=y_test, y=y_pred,
                         labels={'x': 'Real (min)', 'y': 'Previsto (min)'},
                         title='🎯 Modelo CHMSCS - Previsões vs Realidade',
                         color_discrete_sequence=['#7c7adb'])
        fig1.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                 mode='lines', name='Perfeito', line=dict(dash='dash', color='#4dd4c4', width=3)))
        fig1.update_layout(height=520, template='plotly_white')
        
        fig2 = px.bar(importancias, x='Importância (%)', y='Variável', orientation='h',
                     title='📊 Importância dos Fatores',
                     color='Importância (%)', 
                     color_continuous_scale=[[0, '#f5f3ff'], [0.5, '#7c7adb'], [1, '#6b4c9a']],
                     text='Importância (%)')
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2.update_layout(height=420, template='plotly_white')
        
        fig3 = px.bar(analise_esp, x='Quantidade', y='Especialidade',
                     title='📊 Distribuição - CHMSCS', orientation='h', text='Quantidade',
                     color='Média (min)',
                     color_continuous_scale=[[0, '#4dd4c4'], [0.5, '#7c7adb'], [1, '#6b4c9a']])
        fig3.update_traces(textposition='outside')
        fig3.update_layout(height=500, template='plotly_white')
        
        resumo = f"""
<div class="success-banner">
    <h2 style="margin: 0;">✅ Modelo Treinado - Dados CHMSCS!</h2>
    <p style="margin-top: 1rem; font-size: 1.2rem;">{len(df_ml)} cirurgias analisadas</p>
</div>

<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin: 2rem 0;">
    <div class="stat-box"><p class="stat-number">{mae:.1f}</p><p class="stat-label">MAE (min)</p></div>
    <div class="stat-box"><p class="stat-number">{r2*100:.1f}%</p><p class="stat-label">Precisão</p></div>
    <div class="stat-box"><p class="stat-number">{len(df_ml)}</p><p class="stat-label">Cirurgias</p></div>
    <div class="stat-box"><p class="stat-number">{df_ml['Especialidade'].nunique()}</p><p class="stat-label">Especialidades</p></div>
</div>
        """
        
        return resumo, df_ml.head(100), importancias, analise_esp, fig1, fig2, fig3
        
    except Exception as e:
        return f"❌ Erro: {str(e)}", None, None, None, None, None, None

# ============================================
# INTERFACE COMPLETA
# ============================================

with gr.Blocks(
    title="CHMSCS - Sistema Completo de Otimização",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=["Inter", "system-ui", "sans-serif"]
    ),
    css=custom_css
) as app:
    
    # HEADER
    gr.HTML("""
    <div class="header-hospital">
        <div class="logos-container">
            <div class="logo-box">
                <img src="/file/LOGO-COMPLEXO-SAUDE-SCS.jpg" 
                     alt="Complexo de Saúde" style="height: 80px; max-width: 300px;">
            </div>
            <div class="logo-box">
                <img src="/file/LOGO-FUABC.jpg" 
                     alt="FUABC" style="height: 80px; max-width: 200px;">
            </div>
        </div>
        
        <h1 style="font-size: 2.8rem; margin: 1.5rem 0 0.5rem 0; font-weight: 800;">
            Sistema Completo de Otimização Cirúrgica
        </h1>
        
        <p style="font-size: 1.4rem; margin: 0.5rem 0; font-weight: 600;">
            Complexo de Saúde de São Caetano do Sul
        </p>
        
        <p style="font-size: 1.1rem; margin: 1.2rem 0 0 0; opacity: 0.9;">
            🤖 Machine Learning • 🎯 Alocação Automática • 📊 Timeline • 🎮 Simulador • 📥 Exportação
        </p>
    </div>
    """)
    
    with gr.Tabs():
        
        # TAB 1: INÍCIO
        with gr.Tab("🏠 Início"):
            gr.HTML("""
            <div style="max-width: 1100px; margin: 0 auto; text-align: center;">
                <h2 style="color: #6b4c9a; font-size: 2.2rem; margin: 2.5rem 0 1rem 0;">
                    Sistema Completo de Otimização
                </h2>
                <p style="color: #64748b; font-size: 1.3rem; line-height: 1.6;">
                    Plataforma integrada para gestão eficiente do centro cirúrgico<br>
                    <strong style="color: #6b4c9a;">Complexo de Saúde de São Caetano do Sul</strong>
                </p>
                
                <div class="feature-grid">
                    <div class="feature-item">
                        <div style="font-size: 3.5rem;">🤖</div>
                        <h3 style="color: #6b4c9a;">Machine Learning</h3>
                        <p style="color: #64748b;">Previsão com dados reais</p>
                    </div>
                    <div class="feature-item">
                        <div style="font-size: 3.5rem;">📊</div>
                        <h3 style="color: #6b4c9a;">Análise + Gaps</h3>
                        <p style="color: #64748b;">Unificado em 1 clique</p>
                    </div>
                    <div class="feature-item">
                        <div style="font-size: 3.5rem;">🎯</div>
                        <h3 style="color: #6b4c9a;">Slots</h3>
                        <p style="color: #64748b;">Visualização por hora</p>
                    </div>
                    <div class="feature-item">
                        <div style="font-size: 3.5rem;">🔄</div>
                        <h3 style="color: #6b4c9a;">Alocação Auto</h3>
                        <p style="color: #64748b;">Otimização automática</p>
                    </div>
                    <div class="feature-item">
                        <div style="font-size: 3.5rem;">📅</div>
                        <h3 style="color: #6b4c9a;">Timeline</h3>
                        <p style="color: #64748b;">Visualização Gantt</p>
                    </div>
                    <div class="feature-item">
                        <div style="font-size: 3.5rem;">🎮</div>
                        <h3 style="color: #6b4c9a;">Simulador</h3>
                        <p style="color: #64748b;">Teste cenários</p>
                    </div>
                    <div class="feature-item">
                        <div style="font-size: 3.5rem;">📥</div>
                        <h3 style="color: #6b4c9a;">Exportação</h3>
                        <p style="color: #64748b;">Download Excel</p>
                    </div>
                </div>
                
                <div class="metric-card" style="text-align: left; max-width: 800px; margin: 3rem auto;">
                    <h3 style="color: #6b4c9a; margin-top: 0;">🚀 Guia Rápido</h3>
                    <ol style="font-size: 1.15rem; line-height: 2.2;">
                        <li>Faça upload do arquivo <strong>DIMENSIONAMENTO-SALAS-CIRURGICAS-E-ESPECIALIDADES.xlsx</strong></li>
                        <li>Escolha a funcionalidade desejada nas abas</li>
                        <li>Execute a análise ou operação</li>
                        <li>Visualize resultados e exporte se necessário!</li>
                    </ol>
                </div>
            </div>
            """)
        
        # TAB 2: ML
        with gr.Tab("🤖 Machine Learning"):
            gr.Markdown("## 🧠 Modelo de Previsão")
            
            file_ml = gr.File(label="📁 Upload: Grade CHMSCS", file_types=[".xlsx"])
            btn_ml = gr.Button("🚀 Treinar Modelo", variant="primary", size="lg")
            
            output_ml = gr.HTML()
            
            with gr.Row():
                tabela_ml_dados = gr.Dataframe(label="Dados Extraídos")
                tabela_ml_import = gr.Dataframe(label="Importância")
            
            tabela_ml_esp = gr.Dataframe(label="Análise por Especialidade")
            
            with gr.Row():
                grafico_ml1 = gr.Plot()
                grafico_ml2 = gr.Plot()
            
            grafico_ml3 = gr.Plot()
            
            btn_ml.click(fn=treinar_modelo_chmscs, inputs=[file_ml],
                        outputs=[output_ml, tabela_ml_dados, tabela_ml_import, tabela_ml_esp,
                                grafico_ml1, grafico_ml2, grafico_ml3])
        
        # TAB 3: ANÁLISE + GAPS
        with gr.Tab("📊 Análise + Gaps"):
            gr.Markdown("## 📈 Análise Completa Unificada")
            
            file_analise = gr.File(label="📁 Upload: Grade", file_types=[".xlsx"])
            btn_analise = gr.Button("🔍 Analisar Grade + Gaps", variant="primary", size="lg")
            
            output_analise = gr.HTML()
            
            with gr.Row():
                tabela_dist = gr.Dataframe(label="Distribuição")
                tabela_gaps = gr.Dataframe(label="Gaps")
            
            with gr.Row():
                grafico_an1 = gr.Plot()
                grafico_an2 = gr.Plot()
            
            grafico_an3 = gr.Plot()
            
            btn_analise.click(fn=analisar_grade_e_gaps, inputs=[file_analise],
                            outputs=[output_analise, tabela_dist, tabela_gaps, 
                                    grafico_an1, grafico_an2, grafico_an3])
        
        # TAB 4: SLOTS
        with gr.Tab("🎯 Grade com Slots"):
            gr.Markdown("## 🗓️ Visualização por Slots Horários")
            
            gr.HTML("""
            <div class="metric-card">
                <h3 style="color: #6b4c9a; margin-top: 0;">💡 Sistema de Slots</h3>
                <p style="font-size: 1.1rem;">
                    <strong>1 slot = 60 minutos</strong><br>
                    🟢 Pequeno: 1 slot | 🟡 Médio: 2 slots | 🔴 Grande: 3 slots
                </p>
            </div>
            """)
            
            file_slots = gr.File(label="📁 Upload: Grade", file_types=[".xlsx"])
            btn_slots = gr.Button("🎯 Gerar Slots", variant="primary", size="lg")
            
            output_slots = gr.HTML()
            tabela_slots = gr.Dataframe(label="Grade Visual")
            tabela_cap = gr.Dataframe(label="Capacidade")
            grafico_slots = gr.Plot()
            
            btn_slots.click(fn=criar_grade_com_slots, inputs=[file_slots],
                          outputs=[output_slots, tabela_slots, tabela_cap, grafico_slots])
        
        # TAB 5: A) ALOCAÇÃO AUTOMÁTICA
        with gr.Tab("🔄 A) Alocação Automática"):
            gr.Markdown("## 🎯 Alocação Automática por Slots")
            
            gr.HTML("""
            <div class="metric-card">
                <h3 style="color: #6b4c9a; margin-top: 0;">💡 Como Funciona</h3>
                <p style="font-size: 1.1rem; line-height: 1.7;">
                    O sistema identifica gaps automaticamente e aloca as cirurgias da fila 
                    nos melhores horários disponíveis, otimizando por slots necessários.
                </p>
            </div>
            """)
            
            with gr.Row():
                file_grade_aloc = gr.File(label="📁 Grade Atual", file_types=[".xlsx"])
                file_fila_aloc = gr.File(label="📋 Fila de Cirurgias", file_types=[".xlsx"])
            
            btn_aloc = gr.Button("🚀 Alocar Automaticamente", variant="primary", size="lg")
            
            output_aloc = gr.HTML()
            tabela_aloc = gr.Dataframe(label="Cirurgias Alocadas")
            
            btn_aloc.click(fn=alocar_automaticamente_por_slots,
                          inputs=[file_grade_aloc, file_fila_aloc],
                          outputs=[output_aloc, tabela_aloc])
        
        # TAB 6: B) TIMELINE
        with gr.Tab("📅 B) Timeline Visual"):
            gr.Markdown("## 📊 Visualização Timeline (Gantt)")
            
            gr.HTML("""
            <div class="metric-card">
                <h3 style="color: #6b4c9a; margin-top: 0;">💡 Timeline Interativa</h3>
                <p style="font-size: 1.1rem;">
                    Visualize a grade completa em formato Gantt. Cada barra é uma cirurgia.
                    Espaços vazios são gaps disponíveis para alocação.
                </p>
            </div>
            """)
            
            file_timeline = gr.File(label="📁 Upload: Grade", file_types=[".xlsx"])
            btn_timeline = gr.Button("📅 Gerar Timeline", variant="primary", size="lg")
            
            output_timeline = gr.HTML()
            grafico_timeline = gr.Plot()
            
            btn_timeline.click(fn=criar_timeline_visual,
                             inputs=[file_timeline],
                             outputs=[output_timeline, grafico_timeline])
        
        # TAB 7: C) SIMULADOR
        with gr.Tab("🎮 C) Simulador"):
            gr.Markdown("## 🧪 Simulador de Alocação")
            
            gr.HTML("""
            <div class="metric-card">
                <h3 style="color: #6b4c9a; margin-top: 0;">💡 Teste Cenários</h3>
                <p style="font-size: 1.1rem;">
                    Simule: "E se eu alocar 5 cirurgias pequenas + 3 médias?"
                    O sistema calcula se há capacidade disponível!
                </p>
            </div>
            """)
            
            file_sim = gr.File(label="📁 Upload: Grade Atual", file_types=[".xlsx"])
            
            with gr.Row():
                num_peq = gr.Slider(0, 20, 0, 1, label="🟢 Cirurgias Pequenas (1 slot)")
                num_med = gr.Slider(0, 15, 0, 1, label="🟡 Cirurgias Médias (2 slots)")
                num_grd = gr.Slider(0, 10, 0, 1, label="🔴 Cirurgias Grandes (3 slots)")
            
            btn_sim = gr.Button("🎮 Simular Alocação", variant="primary", size="lg")
            
            output_sim = gr.HTML()
            tabela_sim = gr.Dataframe(label="Detalhamento da Simulação")
            
            btn_sim.click(fn=simular_alocacao,
                         inputs=[file_sim, num_peq, num_med, num_grd],
                         outputs=[output_sim, tabela_sim])
        
        # TAB 8: D) EXPORTAR
        with gr.Tab("📥 D) Exportar"):
            gr.Markdown("## 💾 Exportar Grade Otimizada")
            
            gr.HTML("""
            <div class="metric-card">
                <h3 style="color: #6b4c9a; margin-top: 0;">💡 Exportação</h3>
                <p style="font-size: 1.1rem; line-height: 1.7;">
                    Gera arquivo Excel com a grade otimizada (incluindo novas cirurgias alocadas se houver fila).
                    Pronto para implementação!
                </p>
            </div>
            """)
            
            with gr.Row():
                file_export_grade = gr.File(label="📁 Grade Atual (obrigatório)", file_types=[".xlsx"])
                file_export_fila = gr.File(label="📋 Fila de Cirurgias (opcional)", file_types=[".xlsx"])
            
            btn_export = gr.Button("📥 Gerar Excel Otimizado", variant="primary", size="lg")
            
            output_export = gr.HTML()
            file_download = gr.File(label="⬇️ Download da Grade Otimizada")
            
            btn_export.click(fn=exportar_grade_otimizada,
                           inputs=[file_export_grade, file_export_fila],
                           outputs=[output_export, file_download])
        
        # TAB 9: DOCUMENTAÇÃO
        with gr.Tab("📚 Documentação"):
            gr.HTML("""
            <div style="max-width: 950px; margin: 0 auto;">
                <div style="text-align: center; margin: 3rem 0;">
                    <img src="/file/LOGO-COMPLEXO-SAUDE-SCS.jpg" style="height: 100px; margin-bottom: 2rem;">
                    <h2 style="color: #6b4c9a; font-size: 2.2rem;">
                        Complexo de Saúde de São Caetano do Sul
                    </h2>
                    <img src="/file/LOGO-FUABC.jpg" style="height: 90px; margin-top: 1.5rem;">
                </div>
                
                <div class="metric-card">
                    <h3 style="color: #6b4c9a; margin-top: 0;">📖 Funcionalidades Completas</h3>
                    
                    <h4>🤖 Machine Learning</h4>
                    <p>Modelo Random Forest treinado com dados reais da grade do CHMSCS.</p>
                    
                    <h4>📊 Análise + Gaps (Unificado)</h4>
                    <p>Com um único upload, analisa métricas E identifica oportunidades.</p>
                    
                    <h4>🎯 Grade com Slots</h4>
                    <p>Visualização hora por hora com capacidade por porte (pequeno/médio/grande).</p>
                    
                    <h4>🔄 A) Alocação Automática</h4>
                    <p>Sistema aloca cirurgias da fila automaticamente nos melhores gaps.</p>
                    
                    <h4>📅 B) Timeline Visual</h4>
                    <p>Gráfico Gantt mostrando toda a grade de forma visual e interativa.</p>
                    
                    <h4>🎮 C) Simulador</h4>
                    <p>Teste diferentes cenários: "E se eu alocar X cirurgias?"</p>
                    
                    <h4>📥 D) Exportar</h4>
                    <p>Download da grade otimizada em Excel, pronta para uso.</p>
                </div>
                
                <div class="metric-card">
                    <h3 style="color: #6b4c9a; margin-top: 0;">📋 Sistema de Slots</h3>
                    <ul style="font-size: 1.05rem; line-height: 2;">
                        <li><strong>1 slot = 60 minutos</strong></li>
                        <li>🟢 <strong>Pequeno:</strong> &lt; 120 min = 1-2 slots</li>
                        <li>🟡 <strong>Médio:</strong> 120-240 min = 2-4 slots</li>
                        <li>🔴 <strong>Grande:</strong> > 240 min = 4+ slots</li>
                    </ul>
                </div>
                
                <div style="background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); padding: 2.5rem; border-radius: 16px; text-align: center; margin: 3rem 0;">
                    <h3 style="color: #6b4c9a;">📧 Informações</h3>
                    <p style="color: #475569; font-size: 1.1rem; line-height: 2;">
                        <strong>Complexo de Saúde de São Caetano do Sul</strong><br>
                        Fundação do ABC - Desde 1967<br>
                        Sistema de Otimização com IA<br>
                        Departamento de Tecnologia
                    </p>
                </div>
                
                <div style="text-align: center; color: #94a3b8; margin-top: 2rem;">
                    <p><strong>Versão:</strong> 2.0 COMPLETA | <strong>Status:</strong> <span style="color: #22c55e;">✅ Operacional</span></p>
                </div>
            </div>
            """)
    
    # FOOTER
    gr.HTML("""
    <footer>
        <p style="font-size: 1.1rem; color: #6b4c9a; font-weight: 600;">
            🏥 Complexo de Saúde de São Caetano do Sul - CHMSCS
        </p>
        <p style="font-size: 1rem; color: #64748b;">
            Sistema Completo de Otimização Cirúrgica
        </p>
        <p style="font-size: 0.95rem; color: #94a3b8; margin-top: 1rem;">
            Fundação do ABC • 7 Funcionalidades Integradas • IA + Análise + Exportação
        </p>
    </footer>
    """)

if __name__ == "__main__":
    app.launch()
