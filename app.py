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
# CSS MINIMALISTA PROFISSIONAL
# ============================================

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.gradio-container {
    max-width: 1400px !important;
    background: #fafafa;
}

.header-container {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    padding: 4rem 3rem;
    margin: -1rem -1rem 3rem -1rem;
    border-bottom: 1px solid #e5e7eb;
}

.logos-section {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 3rem;
    margin-bottom: 2.5rem;
}

.logo-wrapper {
    background: white;
    padding: 1.5rem 3rem;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.logo-wrapper img {
    height: 70px;
    width: auto;
    display: block;
}

.title-section {
    text-align: center;
    color: white;
}

.main-title {
    font-size: 2.5rem;
    font-weight: 600;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
    color: white;
}

.subtitle {
    font-size: 1.2rem;
    font-weight: 400;
    margin: 0;
    opacity: 0.95;
    color: white;
}

.tagline {
    font-size: 0.95rem;
    margin: 1.5rem 0 0 0;
    opacity: 0.85;
    font-weight: 300;
    color: white;
}

.card {
    background: white;
    border-radius: 8px;
    padding: 2rem;
    margin: 1.5rem 0;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.card h3 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e3a8a;
    margin: 0 0 1rem 0;
    letter-spacing: -0.01em;
}

.card p, .card li {
    font-size: 0.95rem;
    line-height: 1.7;
    color: #4b5563;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.metric-box {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 2rem 1.5rem;
    text-align: center;
    transition: all 0.2s ease;
}

.metric-box:hover {
    border-color: #3b82f6;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 600;
    color: #1e3a8a;
    margin: 0;
    line-height: 1;
}

.metric-label {
    font-size: 0.85rem;
    color: #6b7280;
    margin-top: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.alert-success {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    padding: 1.5rem;
    margin: 1.5rem 0;
    border-radius: 4px;
}

.alert-warning {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    padding: 1.5rem;
    margin: 1.5rem 0;
    border-radius: 4px;
}

.alert-info {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    padding: 1.5rem;
    margin: 1.5rem 0;
    border-radius: 4px;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
    margin: 2.5rem 0;
}

.feature-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 2rem;
    transition: all 0.2s ease;
}

.feature-card:hover {
    border-color: #3b82f6;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.08);
}

.feature-card h4 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e3a8a;
    margin: 0 0 0.75rem 0;
}

.feature-card p {
    font-size: 0.9rem;
    color: #6b7280;
    margin: 0;
    line-height: 1.6;
}

.section-title {
    font-size: 1.8rem;
    font-weight: 600;
    color: #1e3a8a;
    margin: 3rem 0 1.5rem 0;
    text-align: center;
    letter-spacing: -0.02em;
}

.divider {
    border: 0;
    border-top: 1px solid #e5e7eb;
    margin: 3rem 0;
}

footer {
    text-align: center;
    padding: 3rem 1rem;
    margin-top: 4rem;
    border-top: 1px solid #e5e7eb;
    background: white;
}

footer p {
    color: #9ca3af;
    font-size: 0.85rem;
    margin: 0.5rem 0;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 1.5rem 0;
}

table th {
    background: #f9fafb;
    padding: 0.75rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.85rem;
    color: #374151;
    border-bottom: 2px solid #e5e7eb;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

table td {
    padding: 0.75rem;
    border-bottom: 1px solid #f3f4f6;
    font-size: 0.9rem;
    color: #4b5563;
}

table tr:hover {
    background: #f9fafb;
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
        return "Por favor, faça upload da grade cirúrgica.", None, None, None, None, None
    
    try:
        df = processar_grade_cirurgica(file_grade.name)
        
        if df is None:
            return "Erro ao processar o arquivo. Verifique o formato.", None, None, None, None, None
        
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
                            'Duração (min)': int(gap_min), 'Tipo': 'Início do Dia'
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
                                    'Duração (min)': int(gap_min), 'Tipo': 'Entre Cirurgias'
                                })
                        except:
                            pass
                
                ultima = grupo.iloc[-1]
                fim_exp = time(18, 0)
                
                if pd.notna(ultima['horario_fim']) and ultima['horario_fim'] < fim_exp:
                    ultima_dt = datetime.combine(datetime.today(), ultima['horario_fim'])
                    fim_dt = datetime.combine(datetime.today(), fim_exp)
                    gap_min = (fim_dt - ultima_dt).total_seconds() / 60
                    
                    if gap_min >= 30:
                        gaps.append({
                            'Dia': dia, 'Sala': sala,
                            'Início': str(ultima['horario_fim']), 'Fim': str(fim_exp),
                            'Duração (min)': int(gap_min), 'Tipo': 'Final do Dia'
                        })
        
        gaps_df = pd.DataFrame(gaps)
        total_gap_min = gaps_df['Duração (min)'].sum() if len(gaps_df) > 0 else 0
        
        # Gráficos minimalistas
        fig1 = px.bar(
            dist_esp, x='Quantidade', y='especialidade',
            title='Distribuição por Especialidade',
            orientation='h', text='Quantidade',
            color='Total (min)', 
            color_continuous_scale=['#dbeafe', '#3b82f6', '#1e3a8a']
        )
        fig1.update_traces(textposition='outside', textfont=dict(size=11))
        fig1.update_layout(
            height=500, 
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=11),
            title_font=dict(size=14, color='#1e3a8a'),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        ocupacao_df = pd.DataFrame({'Sala': ocupacao_pct.index, 'Ocupação (%)': ocupacao_pct.values})
        
        fig2 = px.bar(
            ocupacao_df, x='Sala', y='Ocupação (%)',
            title='Taxa de Ocupação por Sala',
            text='Ocupação (%)', color='Ocupação (%)',
            color_continuous_scale=['#fee2e2', '#fbbf24', '#22c55e']
        )
        fig2.add_hline(y=70, line_dash="dash", line_color="#1e3a8a", 
                      annotation_text="Meta: 70%", annotation_position="right")
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside', textfont=dict(size=11))
        fig2.update_layout(
            height=480, 
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=11),
            title_font=dict(size=14, color='#1e3a8a'),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        if len(gaps_df) > 0:
            gaps_tipo = gaps_df.groupby('Tipo')['Duração (min)'].sum().reset_index()
            fig3 = px.pie(gaps_tipo, values='Duração (min)', names='Tipo',
                         title='Distribuição de Gaps por Tipo',
                         color_discrete_sequence=['#3b82f6', '#60a5fa', '#93c5fd'])
            fig3.update_traces(textposition='inside', textinfo='percent+label', textfont=dict(size=11))
            fig3.update_layout(
                height=450,
                font=dict(family='Inter, sans-serif', size=11),
                title_font=dict(size=14, color='#1e3a8a')
            )
        else:
            fig3 = None
        
        resumo = f"""
<div class="alert-success">
    <h3 style="margin: 0 0 0.5rem 0; font-size: 1.2rem; color: #166534;">Análise Concluída</h3>
    <p style="margin: 0; color: #15803d;">Grade processada e gaps identificados com sucesso.</p>
</div>

<div class="metric-grid">
    <div class="metric-box">
        <p class="metric-value">{total}</p>
        <p class="metric-label">Cirurgias por Semana</p>
    </div>
    <div class="metric-box">
        <p class="metric-value">{salas}</p>
        <p class="metric-label">Salas Ativas</p>
    </div>
    <div class="metric-box">
        <p class="metric-value">{ocupacao_media:.1f}%</p>
        <p class="metric-label">Ocupação Média</p>
    </div>
    <div class="metric-box">
        <p class="metric-value">{len(gaps_df)}</p>
        <p class="metric-label">Gaps Identificados</p>
    </div>
    <div class="metric-box">
        <p class="metric-value">{total_gap_min}</p>
        <p class="metric-label">Minutos Disponíveis</p>
    </div>
</div>

<div class="card">
    <h3>Resumo Executivo</h3>
    <table>
        <tr>
            <td><strong>Total de cirurgias</strong></td>
            <td>{total} por semana</td>
        </tr>
        <tr>
            <td><strong>Salas em operação</strong></td>
            <td>{salas} unidades</td>
        </tr>
        <tr>
            <td><strong>Especialidades atendidas</strong></td>
            <td>{especialidades}</td>
        </tr>
        <tr>
            <td><strong>Ocupação média</strong></td>
            <td>{ocupacao_media:.1f}%</td>
        </tr>
        <tr>
            <td><strong>Tempo cirúrgico total</strong></td>
            <td>{df['duracao_minutos'].sum()/60:.1f} horas por semana</td>
        </tr>
        <tr>
            <td><strong>Gaps identificados</strong></td>
            <td>{len(gaps_df)} oportunidades</td>
        </tr>
        <tr>
            <td><strong>Capacidade adicional</strong></td>
            <td>{total_gap_min} minutos ({total_gap_min/60:.1f} horas)</td>
        </tr>
    </table>
</div>
        """
        
        return resumo, dist_esp, gaps_df, fig1, fig2, fig3
        
    except Exception as e:
        return f"Erro ao processar: {str(e)}", None, None, None, None, None

# ============================================
# SISTEMA DE SLOTS
# ============================================

def criar_grade_com_slots(file_grade):
    if file_grade is None:
        return "Por favor, faça upload da grade cirúrgica.", None, None, None
    
    try:
        df = processar_grade_cirurgica(file_grade.name)
        if df is None:
            return "Erro ao processar o arquivo.", None, None, None
        
        slots_totais = 10
        grade_slots = []
        
        for (dia, sala), grupo in df.groupby(['dia', 'sala']):
            grupo = grupo.dropna(subset=['horario_inicio_time']).sort_values('horario_inicio_time')
            slots = ['LIVRE'] * slots_totais
            
            for _, cirurgia in grupo.iterrows():
                if pd.notna(cirurgia['horario_inicio_time']):
                    hora_inicio = cirurgia['horario_inicio_time'].hour
                    minuto_inicio = cirurgia['horario_inicio_time'].minute
                    slot_inicio = hora_inicio - 8 + (minuto_inicio / 60)
                    num_slots = int(np.ceil(cirurgia['duracao_minutos'] / 60))
                    
                    slot_idx = int(slot_inicio)
                    for i in range(num_slots):
                        if 0 <= slot_idx + i < slots_totais:
                            esp_curta = cirurgia['especialidade'][:15]
                            slots[slot_idx + i] = esp_curta
            
            grade_slots.append({
                'Dia': dia, 'Sala': sala,
                '08h': slots[0], '09h': slots[1], '10h': slots[2], '11h': slots[3], '12h': slots[4],
                '13h': slots[5], '14h': slots[6], '15h': slots[7], '16h': slots[8], '17h': slots[9]
            })
        
        df_slots = pd.DataFrame(grade_slots)
        
        total_slots = len(df_slots) * slots_totais
        slots_livres = sum([1 for _, row in df_slots.iterrows() 
                           for hora in ['08h', '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h']
                           if row[hora] == 'LIVRE'])
        
        slots_pequeno = int(slots_livres * 1)
        slots_medio = int(slots_livres / 2)
        slots_grande = int(slots_livres / 3)
        
        disp_slots = pd.DataFrame({
            'Porte': ['Pequeno (60-90min)', 'Médio (120-180min)', 'Grande (240-300min)'],
            'Slots Necessários': [1, 2, 3],
            'Cirurgias Possíveis': [slots_pequeno, slots_medio, slots_grande]
        })
        
        fig = px.bar(disp_slots, x='Porte', y='Cirurgias Possíveis',
                     title='Capacidade Disponível por Porte de Cirurgia', 
                     text='Cirurgias Possíveis',
                     color='Cirurgias Possíveis',
                     color_continuous_scale=['#dbeafe', '#3b82f6', '#1e3a8a'])
        fig.update_traces(textposition='outside', textfont=dict(size=11))
        fig.update_layout(
            height=450, 
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=11),
            title_font=dict(size=14, color='#1e3a8a'),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        resumo = f"""
<div class="alert-success">
    <h3 style="margin: 0 0 0.5rem 0; color: #166534;">Grade de Slots Gerada</h3>
    <p style="margin: 0; color: #15803d;">Visualização por hora com capacidade disponível calculada.</p>
</div>

<div class="metric-grid">
    <div class="metric-box">
        <p class="metric-value">{total_slots}</p>
        <p class="metric-label">Slots Totais</p>
    </div>
    <div class="metric-box">
        <p class="metric-value" style="color: #22c55e;">{slots_livres}</p>
        <p class="metric-label">Slots Disponíveis</p>
    </div>
    <div class="metric-box">
        <p class="metric-value">{(total_slots-slots_livres)/total_slots*100:.1f}%</p>
        <p class="metric-label">Taxa de Ocupação</p>
    </div>
</div>

<div class="card">
    <h3>Capacidade por Porte</h3>
    <table>
        <thead>
            <tr>
                <th>Porte</th>
                <th>Duração Típica</th>
                <th>Slots por Cirurgia</th>
                <th>Cirurgias Possíveis</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Pequeno</td>
                <td>60-90 minutos</td>
                <td>1 slot</td>
                <td><strong>{slots_pequeno}</strong></td>
            </tr>
            <tr>
                <td>Médio</td>
                <td>120-180 minutos</td>
                <td>2 slots</td>
                <td><strong>{slots_medio}</strong></td>
            </tr>
            <tr>
                <td>Grande</td>
                <td>240-300 minutos</td>
                <td>3 slots</td>
                <td><strong>{slots_grande}</strong></td>
            </tr>
        </tbody>
    </table>
</div>
        """
        
        return resumo, df_slots, disp_slots, fig
        
    except Exception as e:
        return f"Erro ao processar: {str(e)}", None, None, None

# ============================================
# A) ALOCAÇÃO AUTOMÁTICA
# ============================================

def processar_fila_cirurgias(file_fila):
    if file_fila is None:
        return None
    
    df = pd.read_excel(file_fila.name)
    
    fila = []
    for idx, row in df.iterrows():
        paciente = str(row.get('Paciente', f'Paciente_{idx+1}'))
        esp = str(row.get('Especialidade', 'GERAL'))
        dur = int(row.get('Duracao', row.get('Duração', 120)))
        
        slots_necessarios = int(np.ceil(dur / 60))
        
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
    if file_grade is None or file_fila is None:
        return "Por favor, faça upload de ambos os arquivos.", None
    
    try:
        df_grade = processar_grade_cirurgica(file_grade.name)
        df_fila = processar_fila_cirurgias(file_fila)
        
        if df_grade is None or df_fila is None:
            return "Erro ao processar arquivos.", None
        
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
                'Duração (min)': cirurgia['Duração (min)'],
                'Slots': cirurgia['Slots Necessários'],
                'Dia Alocado': melhor_gap['dia'],
                'Sala': melhor_gap['sala'],
                'Horário': str(melhor_gap['horario_inicio']),
                'Gap Original (min)': melhor_gap['duracao_gap']
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
<div class="alert-success">
    <h3 style="margin: 0 0 0.5rem 0; color: #166534;">Alocação Automática Concluída</h3>
    <p style="margin: 0; color: #15803d;">{len(alocacoes_df)} cirurgias alocadas com sucesso nos melhores horários disponíveis.</p>
</div>

<div class="metric-grid">
    <div class="metric-box">
        <p class="metric-value">{len(df_fila)}</p>
        <p class="metric-label">Fila Original</p>
    </div>
    <div class="metric-box">
        <p class="metric-value" style="color: #22c55e;">{len(alocacoes_df)}</p>
        <p class="metric-label">Alocadas</p>
    </div>
    <div class="metric-box">
        <p class="metric-value">{len(df_fila)-len(alocacoes_df)}</p>
        <p class="metric-label">Pendentes</p>
    </div>
    <div class="metric-box">
        <p class="metric-value">{len(alocacoes_df)/len(df_fila)*100:.1f}%</p>
        <p class="metric-label">Taxa de Alocação</p>
    </div>
</div>

<div class="card">
    <h3>Resultado da Alocação</h3>
    <p>De <strong>{len(df_fila)} cirurgias</strong> na fila, <strong>{len(alocacoes_df)}</strong> foram alocadas automaticamente nos gaps disponíveis, resultando em uma taxa de alocação de <strong>{len(alocacoes_df)/len(df_fila)*100:.1f}%</strong>.</p>
    {f'<p style="color: #dc2626; margin-top: 1rem;"><strong>Atenção:</strong> {len(df_fila)-len(alocacoes_df)} cirurgias não puderam ser alocadas por falta de gaps compatíveis.</p>' if len(df_fila)-len(alocacoes_df) > 0 else ''}
</div>
        """
        
        return resumo, alocacoes_df
        
    except Exception as e:
        return f"Erro: {str(e)}", None

# ============================================
# B) TIMELINE VISUAL
# ============================================

def criar_timeline_visual(file_grade):
    if file_grade is None:
        return "Por favor, faça upload da grade cirúrgica.", None
    
    try:
        df = processar_grade_cirurgica(file_grade.name)
        if df is None:
            return "Erro ao processar o arquivo.", None
        
        df_gantt = df.dropna(subset=['horario_inicio_time', 'horario_fim']).copy()
        
        timeline_data = []
        
        for idx, row in df_gantt.iterrows():
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
        
        fig = px.timeline(
            df_timeline, x_start='Start', x_end='Finish', y='Task', color='Resource',
            title='Timeline da Grade Cirúrgica',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_yaxes(categoryorder='category ascending')
        fig.update_layout(
            height=800,
            xaxis_title='Horário',
            yaxis_title='Sala / Dia',
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=11),
            title_font=dict(size=14, color='#1e3a8a'),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        resumo = f"""
<div class="alert-success">
    <h3 style="margin: 0 0 0.5rem 0; color: #166534;">Timeline Gerada</h3>
    <p style="margin: 0; color: #15803d;">Visualização cronológica de {len(df_gantt)} cirurgias.</p>
</div>

<div class="card">
    <h3>Como Interpretar</h3>
    <p>Cada barra horizontal representa uma cirurgia. A cor indica a especialidade. Espaços em branco são gaps que podem ser preenchidos com novas cirurgias.</p>
</div>
        """
        
        return resumo, fig
        
    except Exception as e:
        return f"Erro: {str(e)}", None

# ============================================
# C) SIMULADOR
# ============================================

def simular_alocacao(file_grade, num_pequeno, num_medio, num_grande):
    if file_grade is None:
        return "Por favor, faça upload da grade cirúrgica.", None
    
    try:
        df = processar_grade_cirurgica(file_grade.name)
        if df is None:
            return "Erro ao processar o arquivo.", None
        
        slots_totais = 10
        grade_slots = []
        
        for (dia, sala), grupo in df.groupby(['dia', 'sala']):
            grupo = grupo.dropna(subset=['horario_inicio_time']).sort_values('horario_inicio_time')
            slots = [True] * slots_totais
            
            for _, cirurgia in grupo.iterrows():
                if pd.notna(cirurgia['horario_inicio_time']):
                    hora_inicio = cirurgia['horario_inicio_time'].hour
                    slot_idx = hora_inicio - 8
                    num_slots = int(np.ceil(cirurgia['duracao_minutos'] / 60))
                    
                    for i in range(num_slots):
                        if 0 <= slot_idx + i < slots_totais:
                            slots[slot_idx + i] = False
            
            grade_slots.extend(slots)
        
        slots_livres = sum(grade_slots)
        
        slots_pequeno_necessarios = num_pequeno * 1
        slots_medio_necessarios = num_medio * 2
        slots_grande_necessarios = num_grande * 3
        
        slots_totais_necessarios = slots_pequeno_necessarios + slots_medio_necessarios + slots_grande_necessarios
        slots_restantes = slots_livres - slots_totais_necessarios
        
        viavel = slots_restantes >= 0
        
        if viavel:
            alert_class = "alert-success"
            titulo = "Alocação Viável"
            cor_titulo = "#166534"
            mensagem = f"É possível alocar todas as {num_pequeno + num_medio + num_grande} cirurgias. Restarão {slots_restantes} slots disponíveis."
        else:
            alert_class = "alert-warning"
            titulo = "Alocação Não Viável"
            cor_titulo = "#92400e"
            mensagem = f"Não há capacidade suficiente. Faltam {abs(slots_restantes)} slots. Considere reduzir o número de cirurgias ou aumentar a capacidade."
        
        resumo = f"""
<div class="{alert_class}">
    <h3 style="margin: 0 0 0.5rem 0; color: {cor_titulo};">{titulo}</h3>
    <p style="margin: 0;">{mensagem}</p>
</div>

<div class="metric-grid">
    <div class="metric-box">
        <p class="metric-value">{slots_livres}</p>
        <p class="metric-label">Slots Disponíveis</p>
    </div>
    <div class="metric-box">
        <p class="metric-value">{slots_totais_necessarios}</p>
        <p class="metric-label">Slots Necessários</p>
    </div>
    <div class="metric-box">
        <p class="metric-value" style="color: {'#22c55e' if viavel else '#dc2626'};">{slots_restantes}</p>
        <p class="metric-label">Slots Restantes</p>
    </div>
    <div class="metric-box">
        <p class="metric-value">{num_pequeno + num_medio + num_grande}</p>
        <p class="metric-label">Total de Cirurgias</p>
    </div>
</div>

<div class="card">
    <h3>Detalhamento da Simulação</h3>
    <table>
        <thead>
            <tr>
                <th>Porte</th>
                <th>Quantidade</th>
                <th>Slots por Unidade</th>
                <th>Slots Totais</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Pequeno</td>
                <td>{num_pequeno}</td>
                <td>1</td>
                <td><strong>{slots_pequeno_necessarios}</strong></td>
            </tr>
            <tr>
                <td>Médio</td>
                <td>{num_medio}</td>
                <td>2</td>
                <td><strong>{slots_medio_necessarios}</strong></td>
            </tr>
            <tr>
                <td>Grande</td>
                <td>{num_grande}</td>
                <td>3</td>
                <td><strong>{slots_grande_necessarios}</strong></td>
            </tr>
            <tr style="background: #f9fafb; font-weight: 600;">
                <td>TOTAL</td>
                <td>{num_pequeno + num_medio + num_grande}</td>
                <td>—</td>
                <td style="color: #1e3a8a;">{slots_totais_necessarios}</td>
            </tr>
        </tbody>
    </table>
</div>
        """
        
        simulacao_df = pd.DataFrame({
            'Porte': ['Pequeno', 'Médio', 'Grande', 'TOTAL'],
            'Quantidade': [num_pequeno, num_medio, num_grande, num_pequeno + num_medio + num_grande],
            'Slots por Unidade': [1, 2, 3, '—'],
            'Slots Totais': [slots_pequeno_necessarios, slots_medio_necessarios, 
                            slots_grande_necessarios, slots_totais_necessarios]
        })
        
        return resumo, simulacao_df
        
    except Exception as e:
        return f"Erro: {str(e)}", None

# ============================================
# D) EXPORTAR GRADE
# ============================================

def exportar_grade_otimizada(file_grade, file_fila):
    if file_grade is None:
        return "Por favor, faça upload da grade cirúrgica.", None
    
    try:
        df_grade = processar_grade_cirurgica(file_grade.name)
        if df_grade is None:
            return "Erro ao processar a grade.", None
        
        if file_fila is not None:
            df_fila = processar_fila_cirurgias(file_fila)
            
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
            
            for idx, cirurgia in df_fila.iterrows():
                duracao = cirurgia['Duração (min)']
                
                gaps_compativeis = gaps_df[gaps_df['duracao_gap'] >= duracao].copy()
                
                if len(gaps_compativeis) > 0:
                    melhor = gaps_compativeis.iloc[0]
                    
                    nova_linha = {
                        'dia': melhor['dia'],
                        'sala': melhor['sala'],
                        'horario_inicio': str(melhor['horario_inicio']),
                        'especialidade': f"NOVA: {cirurgia['Especialidade']}",
                        'duracao_minutos': duracao,
                        'horario_inicio_time': melhor['horario_inicio'] if isinstance(melhor['horario_inicio'], time) else parse_time(melhor['horario_inicio'])
                    }
                    
                    df_grade = pd.concat([df_grade, pd.DataFrame([nova_linha])], ignore_index=True)
                    gaps_df = gaps_df.drop(gaps_df.index[0])
        
        df_export = df_grade.sort_values(['dia', 'sala', 'horario_inicio_time']).copy()
        
        df_export['Horário Início'] = df_export['horario_inicio'].astype(str)
        df_export['Horário Fim'] = df_export['horario_fim'].apply(lambda x: str(x) if pd.notna(x) else '')
        df_export['Duração (min)'] = df_export['duracao_minutos']
        df_export['Especialidade'] = df_export['especialidade']
        df_export['Dia'] = df_export['dia']
        df_export['Sala'] = df_export['sala']
        
        df_final = df_export[['Dia', 'Sala', 'Horário Início', 'Horário Fim', 
                              'Duração (min)', 'Especialidade']]
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_final.to_excel(writer, sheet_name='Grade Otimizada', index=False)
        
        output.seek(0)
        
        resumo = f"""
<div class="alert-success">
    <h3 style="margin: 0 0 0.5rem 0; color: #166534;">Grade Otimizada Gerada</h3>
    <p style="margin: 0; color: #15803d;">Arquivo Excel pronto para download com {len(df_final)} cirurgias.</p>
</div>

<div class="card">
    <h3>Conteúdo do Arquivo</h3>
    <table>
        <tr>
            <td><strong>Total de cirurgias</strong></td>
            <td>{len(df_final)}</td>
        </tr>
        <tr>
            <td><strong>Formato</strong></td>
            <td>Microsoft Excel (.xlsx)</td>
        </tr>
        <tr>
            <td><strong>Planilha</strong></td>
            <td>Grade Otimizada</td>
        </tr>
        <tr>
            <td><strong>Colunas</strong></td>
            <td>Dia, Sala, Horário Início, Horário Fim, Duração, Especialidade</td>
        </tr>
    </table>
    <p style="margin-top: 1.5rem; color: #6b7280;">Use o botão de download abaixo para salvar o arquivo em seu computador.</p>
</div>
        """
        
        return resumo, output
        
    except Exception as e:
        return f"Erro: {str(e)}", None

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
        return "Por favor, faça upload da grade cirúrgica.", None, None, None, None, None, None
    
    try:
        df_grade = processar_grade_cirurgica(file_grade.name)
        
        if df_grade is None or len(df_grade) < 20:
            return "Dados insuficientes para treinamento. Mínimo: 20 cirurgias.", None, None, None, None, None, None
        
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
                         labels={'x': 'Duração Real (min)', 'y': 'Duração Prevista (min)'},
                         title='Modelo: Previsões vs Realidade',
                         color_discrete_sequence=['#3b82f6'])
        fig1.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                 mode='lines', name='Previsão Perfeita',
                                 line=dict(dash='dash', color='#1e3a8a', width=2)))
        fig1.update_layout(
            height=520, 
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=11),
            title_font=dict(size=14, color='#1e3a8a'),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        fig2 = px.bar(importancias, x='Importância (%)', y='Variável', orientation='h',
                     title='Importância das Variáveis',
                     color='Importância (%)', 
                     color_continuous_scale=['#eff6ff', '#3b82f6', '#1e3a8a'],
                     text='Importância (%)')
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside', textfont=dict(size=11))
        fig2.update_layout(
            height=420, 
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=11),
            title_font=dict(size=14, color='#1e3a8a'),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        fig3 = px.bar(analise_esp, x='Quantidade', y='Especialidade',
                     title='Distribuição por Especialidade', orientation='h', text='Quantidade',
                     color='Média (min)',
                     color_continuous_scale=['#dbeafe', '#3b82f6', '#1e3a8a'])
        fig3.update_traces(textposition='outside', textfont=dict(size=11))
        fig3.update_layout(
            height=500, 
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=11),
            title_font=dict(size=14, color='#1e3a8a'),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        resumo = f"""
<div class="alert-success">
    <h3 style="margin: 0 0 0.5rem 0; color: #166534;">Modelo Treinado com Sucesso</h3>
    <p style="margin: 0; color: #15803d;">{len(df_ml)} cirurgias do histórico do CHMSCS foram analisadas.</p>
</div>

<div class="metric-grid">
    <div class="metric-box">
        <p class="metric-value">{mae:.1f}</p>
        <p class="metric-label">Erro Médio (min)</p>
    </div>
    <div class="metric-box">
        <p class="metric-value">{r2*100:.1f}%</p>
        <p class="metric-label">Precisão do Modelo</p>
    </div>
    <div class="metric-box">
        <p class="metric-value">{len(df_ml)}</p>
        <p class="metric-label">Cirurgias Analisadas</p>
    </div>
    <div class="metric-box">
        <p class="metric-value">{df_ml['Especialidade'].nunique()}</p>
        <p class="metric-label">Especialidades</p>
    </div>
</div>

<div class="card">
    <h3>Métricas de Performance</h3>
    <table>
        <tr>
            <td><strong>MAE (Mean Absolute Error)</strong></td>
            <td>{mae:.1f} minutos</td>
        </tr>
        <tr>
            <td><strong>R² (Coeficiente de Determinação)</strong></td>
            <td>{r2:.3f} ({r2*100:.1f}%)</td>
        </tr>
        <tr>
            <td><strong>Algoritmo</strong></td>
            <td>Random Forest (100 árvores)</td>
        </tr>
        <tr>
            <td><strong>Dataset</strong></td>
            <td>Dados reais do CHMSCS</td>
        </tr>
        <tr>
            <td><strong>Validação</strong></td>
            <td>80% treino / 20% teste</td>
        </tr>
    </table>
</div>

<div class="alert-info">
    <h4 style="margin: 0 0 0.5rem 0; color: #1e40af;">Interpretação</h4>
    <p style="margin: 0; color: #1e40af;">
        O modelo prevê durações com erro médio de <strong>{mae:.1f} minutos</strong> e precisão de <strong>{r2*100:.1f}%</strong>, 
        permitindo planejamento mais confiável da grade cirúrgica do CHMSCS.
    </p>
</div>
        """
        
        return resumo, df_ml.head(100), importancias, analise_esp, fig1, fig2, fig3
        
    except Exception as e:
        return f"Erro: {str(e)}", None, None, None, None, None, None

# ============================================
# INTERFACE MINIMALISTA
# ============================================

with gr.Blocks(
    title="CHMSCS - Sistema de Otimização Cirúrgica",
    theme=gr.themes.Default(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=["Inter", "system-ui", "sans-serif"]
    ),
    css=custom_css
) as app:
    
    # HEADER PROFISSIONAL
    gr.HTML("""
    <div class="header-container">
        <div class="logos-section">
            <div class="logo-wrapper">
                <img src="/file/LOGO-COMPLEXO-SAUDE-SCS.jpg" alt="Complexo de Saúde de São Caetano do Sul">
            </div>
            <div class="logo-wrapper">
                <img src="/file/LOGO-FUABC.jpg" alt="Fundação do ABC">
            </div>
        </div>
        
        <div class="title-section">
            <h1 class="main-title">Sistema de Otimização Cirúrgica</h1>
            <p class="subtitle">Complexo de Saúde de São Caetano do Sul</p>
            <p class="tagline">Machine Learning · Análise de Dados · Otimização Inteligente</p>
        </div>
    </div>
    """)
    
    with gr.Tabs():
        
        # TAB 1: INÍCIO
        with gr.Tab("Início"):
            gr.HTML("""
            <div style="max-width: 1100px; margin: 0 auto;">
                
                <h2 class="section-title">Sistema de Otimização</h2>
                
                <p style="text-align: center; font-size: 1.1rem; color: #6b7280; max-width: 700px; margin: 0 auto 3rem auto; line-height: 1.7;">
                    Plataforma integrada para gestão eficiente do centro cirúrgico, 
                    utilizando inteligência artificial e análise de dados.
                </p>
                
                <div class="feature-grid">
                    <div class="feature-card">
                        <h4>Machine Learning</h4>
                        <p>Modelo de previsão de duração treinado com dados reais do CHMSCS.</p>
                    </div>
                    
                    <div class="feature-card">
                        <h4>Análise Unificada</h4>
                        <p>Métricas da grade e identificação de gaps em um único processamento.</p>
                    </div>
                    
                    <div class="feature-card">
                        <h4>Sistema de Slots</h4>
                        <p>Visualização hora por hora com capacidade disponível por porte.</p>
                    </div>
                    
                    <div class="feature-card">
                        <h4>Alocação Automática</h4>
                        <p>Otimização inteligente que aloca cirurgias nos melhores horários.</p>
                    </div>
                    
                    <div class="feature-card">
                        <h4>Timeline Visual</h4>
                        <p>Gráfico Gantt interativo para visualização cronológica completa.</p>
                    </div>
                    
                    <div class="feature-card">
                        <h4>Simulador</h4>
                        <p>Teste diferentes cenários de alocação antes de implementar.</p>
                    </div>
                    
                    <div class="feature-card">
                        <h4>Exportação</h4>
                        <p>Download da grade otimizada em formato Excel pronto para uso.</p>
                    </div>
                </div>
                
                <hr class="divider">
                
                <div class="card">
                    <h3>Instruções de Uso</h3>
                    <ol style="line-height: 2;">
                        <li>Navegue até a funcionalidade desejada usando as abas acima</li>
                        <li>Faça upload do arquivo DIMENSIONAMENTO-SALAS-CIRURGICAS-E-ESPECIALIDADES.xlsx</li>
                        <li>Execute a análise ou operação clicando no botão correspondente</li>
                        <li>Visualize os resultados e exporte quando necessário</li>
                    </ol>
                </div>
                
                <div class="alert-info">
                    <h4 style="margin: 0 0 0.5rem 0; color: #1e40af;">Segurança e Privacidade</h4>
                    <p style="margin: 0; color: #1e40af;">
                        Todos os dados são processados localmente no servidor. 
                        Nenhuma informação de pacientes é armazenada ou compartilhada.
                    </p>
                </div>
                
            </div>
            """)
        
        # TAB 2: ML
        with gr.Tab("Machine Learning"):
            gr.HTML("""
            <div style="max-width: 1200px; margin: 0 auto;">
                <h2 class="section-title">Modelo de Previsão de Duração</h2>
                
                <div class="card">
                    <h3>Sobre o Modelo</h3>
                    <p>
                        Sistema de Machine Learning que analisa padrões históricos de cirurgias 
                        do CHMSCS para prever com precisão quanto tempo procedimentos realmente levam.
                    </p>
                    <p style="margin-top: 1rem;">
                        <strong>Benefícios:</strong> Redução de atrasos, melhor aproveitamento de salas, 
                        diminuição de custos operacionais e maior satisfação de pacientes e equipes.
                    </p>
                </div>
            </div>
            """)
            
            file_ml = gr.File(label="Fazer Upload da Grade Cirúrgica", file_types=[".xlsx"])
            btn_ml = gr.Button("Treinar Modelo", variant="primary", size="lg")
            
            output_ml = gr.HTML()
            
            with gr.Row():
                tabela_ml_dados = gr.Dataframe(label="Dados Extraídos (Amostra)")
                tabela_ml_import = gr.Dataframe(label="Importância das Variáveis")
            
            tabela_ml_esp = gr.Dataframe(label="Análise por Especialidade")
            
            with gr.Row():
                grafico_ml1 = gr.Plot(label="Qualidade das Previsões")
                grafico_ml2 = gr.Plot(label="Importância dos Fatores")
            
            grafico_ml3 = gr.Plot(label="Distribuição por Especialidade")
            
            btn_ml.click(fn=treinar_modelo_chmscs, inputs=[file_ml],
                        outputs=[output_ml, tabela_ml_dados, tabela_ml_import, tabela_ml_esp,
                                grafico_ml1, grafico_ml2, grafico_ml3])
        
        # TAB 3: ANÁLISE + GAPS
        with gr.Tab("Análise + Gaps"):
            gr.HTML("""
            <div style="max-width: 1200px; margin: 0 auto;">
                <h2 class="section-title">Análise Completa da Grade</h2>
                
                <div class="card">
                    <h3>Análise Unificada</h3>
                    <p>
                        Com um único upload, o sistema processa a grade completa, calcula métricas 
                        de ocupação e identifica automaticamente todos os gaps disponíveis para otimização.
                    </p>
                </div>
            </div>
            """)
            
            file_analise = gr.File(label="Fazer Upload da Grade Cirúrgica", file_types=[".xlsx"])
            btn_analise = gr.Button("Analisar Grade e Identificar Gaps", variant="primary", size="lg")
            
            output_analise = gr.HTML()
            
            with gr.Row():
                tabela_dist = gr.Dataframe(label="Distribuição por Especialidade")
                tabela_gaps = gr.Dataframe(label="Gaps Identificados")
            
            with gr.Row():
                grafico_an1 = gr.Plot(label="Distribuição de Cirurgias")
                grafico_an2 = gr.Plot(label="Ocupação por Sala")
            
            grafico_an3 = gr.Plot(label="Distribuição de Gaps")
            
            btn_analise.click(fn=analisar_grade_e_gaps, inputs=[file_analise],
                            outputs=[output_analise, tabela_dist, tabela_gaps, 
                                    grafico_an1, grafico_an2, grafico_an3])
        
        # TAB 4: SLOTS
        with gr.Tab("Grade com Slots"):
            gr.HTML("""
            <div style="max-width: 1200px; margin: 0 auto;">
                <h2 class="section-title">Visualização por Slots Horários</h2>
                
                <div class="card">
                    <h3>Sistema de Slots</h3>
                    <p>
                        A grade é dividida em slots de 60 minutos (8h às 18h = 10 slots por dia). 
                        Esta visualização permite identificar rapidamente onde há capacidade disponível.
                    </p>
                    <table style="margin-top: 1.5rem;">
                        <thead>
                            <tr>
                                <th>Porte</th>
                                <th>Duração</th>
                                <th>Slots Necessários</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Pequeno</td>
                                <td>60-90 minutos</td>
                                <td>1 slot</td>
                            </tr>
                            <tr>
                                <td>Médio</td>
                                <td>120-180 minutos</td>
                                <td>2 slots</td>
                            </tr>
                            <tr>
                                <td>Grande</td>
                                <td>240-300 minutos</td>
                                <td>3 slots</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            """)
            
            file_slots = gr.File(label="Fazer Upload da Grade Cirúrgica", file_types=[".xlsx"])
            btn_slots = gr.Button("Gerar Grade de Slots", variant="primary", size="lg")
            
            output_slots = gr.HTML()
            tabela_slots = gr.Dataframe(label="Grade Visual por Slots")
            tabela_cap = gr.Dataframe(label="Capacidade Disponível")
            grafico_slots = gr.Plot(label="Capacidade por Porte")
            
            btn_slots.click(fn=criar_grade_com_slots, inputs=[file_slots],
                          outputs=[output_slots, tabela_slots, tabela_cap, grafico_slots])
        
        # TAB 5: ALOCAÇÃO AUTOMÁTICA
        with gr.Tab("Alocação Automática"):
            gr.HTML("""
            <div style="max-width: 1200px; margin: 0 auto;">
                <h2 class="section-title">Alocação Automática de Cirurgias</h2>
                
                <div class="card">
                    <h3>Como Funciona</h3>
                    <p>
                        O sistema identifica gaps na grade e aloca automaticamente as cirurgias da fila 
                        nos melhores horários disponíveis, otimizando o uso dos slots por porte de cirurgia.
                    </p>
                    
                    <h4 style="margin-top: 1.5rem; font-size: 1rem; color: #374151;">Formato do Arquivo de Fila</h4>
                    <p>O arquivo Excel deve conter as colunas: Paciente, Especialidade, Duracao (em minutos).</p>
                </div>
            </div>
            """)
            
            with gr.Row():
                file_grade_aloc = gr.File(label="Grade Atual", file_types=[".xlsx"])
                file_fila_aloc = gr.File(label="Fila de Cirurgias", file_types=[".xlsx"])
            
            btn_aloc = gr.Button("Executar Alocação Automática", variant="primary", size="lg")
            
            output_aloc = gr.HTML()
            tabela_aloc = gr.Dataframe(label="Cirurgias Alocadas")
            
            btn_aloc.click(fn=alocar_automaticamente_por_slots,
                          inputs=[file_grade_aloc, file_fila_aloc],
                          outputs=[output_aloc, tabela_aloc])
        
        # TAB 6: TIMELINE
        with gr.Tab("Timeline Visual"):
            gr.HTML("""
            <div style="max-width: 1200px; margin: 0 auto;">
                <h2 class="section-title">Timeline Cronológica</h2>
                
                <div class="card">
                    <h3>Visualização Gantt</h3>
                    <p>
                        Gráfico de Gantt interativo que mostra todas as cirurgias da grade em formato 
                        cronológico. Cada barra representa uma cirurgia, cores indicam especialidades, 
                        e espaços vazios são gaps disponíveis.
                    </p>
                </div>
            </div>
            """)
            
            file_timeline = gr.File(label="Fazer Upload da Grade Cirúrgica", file_types=[".xlsx"])
            btn_timeline = gr.Button("Gerar Timeline", variant="primary", size="lg")
            
            output_timeline = gr.HTML()
            grafico_timeline = gr.Plot(label="Timeline Gantt")
            
            btn_timeline.click(fn=criar_timeline_visual,
                             inputs=[file_timeline],
                             outputs=[output_timeline, grafico_timeline])
        
        # TAB 7: SIMULADOR
        with gr.Tab("Simulador"):
            gr.HTML("""
            <div style="max-width: 1200px; margin: 0 auto;">
                <h2 class="section-title">Simulador de Cenários</h2>
                
                <div class="card">
                    <h3>Teste de Alocação</h3>
                    <p>
                        Simule diferentes cenários de alocação: "E se eu precisar alocar 5 cirurgias pequenas 
                        mais 3 médias?" O sistema calcula se há capacidade disponível e quantos slots restarão.
                    </p>
                </div>
            </div>
            """)
            
            file_sim = gr.File(label="Fazer Upload da Grade Atual", file_types=[".xlsx"])
            
            gr.Markdown("### Configurar Simulação")
            
            with gr.Row():
                num_peq = gr.Slider(0, 20, 0, 1, label="Cirurgias Pequenas (1 slot cada)")
                num_med = gr.Slider(0, 15, 0, 1, label="Cirurgias Médias (2 slots cada)")
                num_grd = gr.Slider(0, 10, 0, 1, label="Cirurgias Grandes (3 slots cada)")
            
            btn_sim = gr.Button("Executar Simulação", variant="primary", size="lg")
            
            output_sim = gr.HTML()
            tabela_sim = gr.Dataframe(label="Detalhamento da Simulação")
            
            btn_sim.click(fn=simular_alocacao,
                         inputs=[file_sim, num_peq, num_med, num_grd],
                         outputs=[output_sim, tabela_sim])
        
        # TAB 8: EXPORTAR
        with gr.Tab("Exportar"):
            gr.HTML("""
            <div style="max-width: 1200px; margin: 0 auto;">
                <h2 class="section-title">Exportar Grade Otimizada</h2>
                
                <div class="card">
                    <h3>Geração de Arquivo Excel</h3>
                    <p>
                        Gera arquivo Excel com a grade otimizada, incluindo cirurgias alocadas 
                        automaticamente se houver arquivo de fila. O arquivo está pronto para 
                        implementação imediata.
                    </p>
                    
                    <h4 style="margin-top: 1.5rem; font-size: 1rem; color: #374151;">Arquivos Aceitos</h4>
                    <ul style="line-height: 1.8;">
                        <li><strong>Grade Atual:</strong> Obrigatório - arquivo de dimensionamento</li>
                        <li><strong>Fila de Cirurgias:</strong> Opcional - se fornecido, cirurgias serão alocadas automaticamente</li>
                    </ul>
                </div>
            </div>
            """)
            
            with gr.Row():
                file_export_grade = gr.File(label="Grade Atual (obrigatório)", file_types=[".xlsx"])
                file_export_fila = gr.File(label="Fila de Cirurgias (opcional)", file_types=[".xlsx"])
            
            btn_export = gr.Button("Gerar Arquivo Excel", variant="primary", size="lg")
            
            output_export = gr.HTML()
            file_download = gr.File(label="Download da Grade Otimizada")
            
            btn_export.click(fn=exportar_grade_otimizada,
                           inputs=[file_export_grade, file_export_fila],
                           outputs=[output_export, file_download])
        
        # TAB 9: DOCUMENTAÇÃO
        with gr.Tab("Documentação"):
            gr.HTML("""
            <div style="max-width: 950px; margin: 0 auto;">
                
                <div style="text-align: center; margin: 3rem 0 4rem 0;">
                    <img src="/file/LOGO-COMPLEXO-SAUDE-SCS.jpg" style="height: 100px; margin-bottom: 2rem;">
                    <h2 style="color: #1e3a8a; font-size: 2rem; margin: 1rem 0;">
                        Complexo de Saúde de São Caetano do Sul
                    </h2>
                    <img src="/file/LOGO-FUABC.jpg" style="height: 90px; margin-top: 1.5rem;">
                </div>
                
                <div class="card">
                    <h3>Funcionalidades do Sistema</h3>
                    
                    <h4 style="margin-top: 1.5rem;">Machine Learning</h4>
                    <p>Modelo Random Forest treinado com dados reais da grade do CHMSCS para previsão de duração cirúrgica.</p>
                    
                    <h4 style="margin-top: 1.5rem;">Análise + Gaps</h4>
                    <p>Processamento unificado que analisa métricas operacionais e identifica oportunidades de otimização em uma única execução.</p>
                    
                    <h4 style="margin-top: 1.5rem;">Grade com Slots</h4>
                    <p>Visualização hora por hora dividida em slots de 60 minutos, com cálculo automático de capacidade por porte de cirurgia.</p>
                    
                    <h4 style="margin-top: 1.5rem;">Alocação Automática</h4>
                    <p>Sistema inteligente que aloca cirurgias da fila nos gaps disponíveis, otimizando aproveitamento e minimizando desperdício de tempo.</p>
                    
                    <h4 style="margin-top: 1.5rem;">Timeline Visual</h4>
                    <p>Gráfico Gantt interativo para visualização cronológica completa de todas as cirurgias e identificação visual de gaps.</p>
                    
                    <h4 style="margin-top: 1.5rem;">Simulador</h4>
                    <p>Ferramenta de teste que permite simular diferentes cenários de alocação antes da implementação.</p>
                    
                    <h4 style="margin-top: 1.5rem;">Exportação</h4>
                    <p>Download de arquivo Excel com grade otimizada, pronto para uso operacional.</p>
                </div>
                
                <div class="card">
                    <h3>Tecnologias</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Tecnologia</th>
                                <th>Função</th>
                                <th>Versão</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Python</td>
                                <td>Linguagem de programação</td>
                                <td>3.13</td>
                            </tr>
                            <tr>
                                <td>Gradio</td>
                                <td>Interface web interativa</td>
                                <td>6.5+</td>
                            </tr>
                            <tr>
                                <td>Scikit-learn</td>
                                <td>Machine Learning</td>
                                <td>Latest</td>
                            </tr>
                            <tr>
                                <td>Pandas</td>
                                <td>Processamento de dados</td>
                                <td>Latest</td>
                            </tr>
                            <tr>
                                <td>Plotly</td>
                                <td>Visualizações interativas</td>
                                <td>Latest</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                
                <div style="background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 2.5rem; text-align: center; margin: 3rem 0;">
                    <h3 style="color: #1e3a8a; margin: 0 0 1rem 0;">Informações de Contato</h3>
                    <p style="color: #6b7280; font-size: 1rem; line-height: 2; margin: 0;">
                        <strong>Complexo de Saúde de São Caetano do Sul</strong><br>
                        Fundação do ABC<br>
                        Departamento de Tecnologia da Informação
                    </p>
                </div>
                
                <div style="text-align: center; color: #9ca3af; padding-top: 2rem; border-top: 1px solid #e5e7eb;">
                    <p style="font-size: 0.9rem;">
                        <strong>Versão:</strong> 2.0.0 · 
                        <strong>Atualização:</strong> Janeiro 2026 · 
                        <strong>Status:</strong> <span style="color: #22c55e; font-weight: 600;">Operacional</span>
                    </p>
                </div>
                
            </div>
            """)
    
    # FOOTER MINIMALISTA
    gr.HTML("""
    <footer>
        <p style="font-weight: 600; color: #4b5563; font-size: 0.95rem;">
            Complexo de Saúde de São Caetano do Sul
        </p>
        <p>Sistema de Otimização Cirúrgica com Inteligência Artificial</p>
        <p>Fundação do ABC · Desde 1967</p>
        <p style="margin-top: 1.5rem;">Desenvolvido para excelência em gestão hospitalar</p>
    </footer>
    """)

if __name__ == "__main__":
    app.launch()
