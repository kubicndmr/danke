import sdg_helper

from num2words import num2words


class SDGPrompts:
    def __init__(self):
        pass


    def init_personal_prompts(self):
        self.system_radiologe = f"Du bist ein hilfsbereiter Assistent, der realistische Gespräche führt, indem er eine bestimmte Persona simuliert. Die Persona gehört zu einem Radiologen, der bei einem Krankenhaus in Deutschland arbeitet. Du simulierst {self.radiologe}"
        self.system_assistant = f"Du bist hilfsbereiter Assistent, der realistische Gespräche führt, indem er eine bestimmte Persona simuliert. Die Persona gehört zu einem medizinischen Assistenten, der bei einem Krankenhaus in Deutschland arbeitet. Du simulierst {self.assistent}"
        self.system_patient = f"Du bist ein hilfreicher Assistent, der realistische Gespräche führt, indem er eine bestimmte Persona simuliert. Die Persona gehört zu einem Patient, derin einem Krankenhaus in Deutschland operiert wird. Du simulierst {self.patient}"


    def init_OR(self):
        self.radiologe = sdg_helper.sample_radiologe()
        self.assistent = sdg_helper.sample_assistent()
        self.patient = sdg_helper.sample_patient()
        self.topic = sdg_helper.sample_daily_topic()
        self.problem = sdg_helper.sample_problem()
        self.init_personal_prompts()


    def get_step_description(self, step_label):
        if step_label in sdg_helper.surgical_steps[0]:
            return """
    - Schritt 0.1 Positionierung des Patienten auf dem Tisch: Der Patient wird in eine stabile, komfortable Position gebracht, in Rückenlage.
    - Schritt 0.2 Tisch fährt hoch: Der Operationstisch wird auf eine ergonomische Höhe für den Radiologen und Assistenten gebracht, um den Eingriff effizient durchzuführen.
    - Schritt 0.3 Radiologe sterilisiert sich: Der Radiologe, der die Operation übernimmt, bereitet sich durch Sterilisation und Anziehen steriler Kleidung vor, um Infektionen zu verhindern.
    - Schritt 0.4 Vorbereitung des sterilen Materials: Alle benötigten Materialien (Katheter, Nadeln, Nahtmaterial, etc.) werden in einem sterilen Feld vorbereitet, um eine Kontamination zu vermeiden.
    - Schritt 0.5 Patient steril abgedeckt: Der Patient wird steril abgedeckt, damit nur der operative Bereich freiliegt. Dies reduziert das Risiko von Infektionen."""
        elif step_label in sdg_helper.surgical_steps[1]:
            return """
    - Schritt 1.1 Lokale Anästhesie: Das operative Gebiet wird lokal betäubt, um den Patienten während des Eingriffs schmerzfrei zu halten.
    - Schritt 1.2 Ultraschallgeführte Punktion: Mit Ultraschall wird die Zielvene (meist die Vena subclavia oder die Vena jugularis interna) angesteuert und punktiert. Diese Methode verbessert die Sicherheit und Präzision, da der Zugang zur Vene sichtbar kontrolliert wird."""
        elif step_label in sdg_helper.surgical_steps[2]:
            return """
    - Schritt 2.1 Röntgenmaschine fährt ein: Die Röntgenmaschine wird in Position gebracht, um die nachfolgenden Schritte bildgebend zu überwachen.
    - Schritt 2.2 Durchleuchtung im Bereich der Subklavia: Eine Durchleuchtung stellt sicher, dass der Führungsdraht korrekt in die Vene eingeführt wird.
    - Schritt 2.3 Durchleuchtung im Bereich der Vena cava inferior (VCI): Der Führungsdraht wird bis zur Vena cava inferior vorgeschoben, um die spätere Position des Katheters zu bestätigen.
    - Schritt 2.4 Röntgenmaschine fährt heraus: Nach erfolgreicher Platzierung des Drahtes wird die Röntgenmaschine für den nächsten Schritt zurückgezogen."""
        elif step_label in sdg_helper.surgical_steps[3]:
            return """
    - Schritt 3.1 Lokale Anästhesie: Das Gebiet über dem Schlüsselbein wird lokal betäubt, um den Schnitt für die Portkammer vorzubereiten.
    - Schritt 3.2 Inzision: Es wird ein Hautschnitt durchgeführt, um Zugang zum Unterhautgewebe zu schaffen, wo der Port später platziert wird.
    - Schritt 3.3 Pouch-Vorbereitung: Mit einem stumpfen Präparierinstrument wird eine kleine Tasche im Gewebe geschaffen, um Platz für die Portkammer zu machen.
    - Schritt 3.4 Hülleplatzierung: Eine Hülse (Schleuse) wird um den Führungsdraht gelegt, um den Katheter über den Draht in die Vene einzuführen."""
        elif step_label in sdg_helper.surgical_steps[4]:
            return """
    - Shritt 4.1 Röntgenmaschine fährt ein: Die Röntgenmaschine wird erneut aktiviert, um die Platzierung des Katheters zu überwachen.
    - Shritt 4.2 Durchleuchtung des VCI-Bereichs: Die Position des Katheters in der Vena cava inferior wird überprüft, um sicherzustellen, dass er korrekt platziert ist.
    - Shritt 4.3 Positionierung des Katheters: Der Katheter wird bis in die richtige Tiefe vorgeschoben, meist knapp oberhalb der rechten Herzvorhofgrenze."""
        elif step_label in sdg_helper.surgical_steps[5]:
            return """
    - Schritt 5.1 Kürzen des Katheters: Der Katheter wird auf die richtige Länge gekürzt, um eine optimale Funktion und Platzierung zu gewährleisten.
    - Schritt 5.2 Röntgenmaschine fährt aus: Die Bildgebung wird abgeschlossen, da die endgültige Katheterplatzierung gesichert ist.
    - Schritt 5.3 Anschluss des Katheters an die Portkapsel: Der Katheter wird mit der Portkapsel verbunden, die das Medikament später in den Blutkreislauf leitet.
    - Schritt 5.4 Positionierung der Portkapsel im Pouch: Die Portkapsel wird in den zuvor geschaffenen Pouch implantiert und fixiert.
    - Schritt 5.5 Chirurgische Naht: Der Hautschnitt wird in mehreren Schichten vernäht, um die Implantationsstelle zu verschließen.
    - Schritt 5.6 Punktion der Portkapsel: Zum Testen der Portfunktion wird die Kapsel punktiert, um sicherzustellen, dass alles korrekt verbunden ist."""
        elif step_label in sdg_helper.surgical_steps[6]:
            return """
    - Schritt 6.1 Röntgenmaschine fährt ein: Die Röntgenmaschine wird aktiviert, um die Funktion des Katheters zu überprüfen.
    - Schritt 6.2 Digitale Subtraktionsangiographie des Brustbereichs: Eine Kontrastmittelgabe mit digitaler Subtraktionsangiographie stellt sicher, dass der Katheter durchgängig und richtig platziert ist.
    - Schritt 6.3 Röntgenmaschine fährt in Parkposition aus: Nach der abschließenden Überprüfung wird die Röntgenmaschine endgültig deaktiviert."""
        elif step_label in sdg_helper.surgical_steps[7]:
            return """
    - Schritt 7.1 Steriles Pflaster auflegen: Über der Naht wird ein steriles Pflaster angebracht, um die Wunde zu schützen.
    - Schritt 7.2 Tisch fährt nach unten: Der Operationstisch wird abgesenkt, um den Patienten sicher vom Tisch zu transferieren."""
        elif step_label == 'Alltäglich':
            pass
        else:
            raise ValueError('Etwas ist schiefgelaufen')


    def get_radiologe_prompt(self, iteration, step_df, answer_df, step_label, step_count, n_examples=3, n_context=25):
        prompt = f"""{self.system_radiologe}
        
Das Ziel ist es, realistische, hochwertige, einzigartige, und medizinisch korrekte Gespräche in einem Operationssaal während einer Port-Katheter-Platzierung zu simulieren.

* Personal: Die Port-Katheter-Platzierung wird von einem Radiologen und einem medizinischen Assistenten in der radiologischen Abteilung durchgeführt. Der Radiologe ist verantwortlich für die Durchführung des Verfahrens, die Kommunikation mit dem Assistenten, um Anweisungen zu geben, und die Interaktion mit dem Patienten, um dessen Zustand zu überwachen und ihn ruhig zu halten. Der Assistent ist für die Vorbereitung steriler Materialien und die Bedienung des Röntgengeräts auf Anweisung des Radiologen zuständig. Der Patient ist die Person, die sich dem Verfahren unterzieht. Heute befinden sich die Patientin/der Patient <{self.patient}> und die Assistentin/der Assistent <{self.assistent}> im OP.

* Operation: Chirurgische Phasen und chirurgische Schritte darstellen eine typische Operation. Die Phasen beziehen sich auf die großen Abschnitte des Verfahrens, in denen die wichtigsten Schritte beschrieben werden. Chirurgische Schritte sind die spezifischen Aufgaben, die innerhalb jeder Phase ausgeführt werden sollen. Die Phasen der Port-Katheter-Platzierungsoperation sind folgende:
    - Phase 0 Vorbereitung: In dieser Phase wird der Patient vorbereitet, das Operationsgebiet sterilisiert, und das notwendige Equipment bereitgestellt.
    - Phase 1 Punktion: Der Radiologe punktiert die Haut und das darunterliegende Gewebe, um Zugang zur Vene zu erhalten.
    - Phase 2 Führungsdraht: Ein Führungsdraht wird vorsichtig durch die Punktionsstelle in die Vene eingeführt, um den Katheterweg zu sichern.
    - Phase 3 Pouchvorbereitung und Katheterplatzierung: Ein subkutaner Pouch wird vorbereitet, und der Port-Katheter wird an die richtige Position gebracht.
    - Phase 4 Katheterpositionierung: Der Katheter wird endgültig positioniert, sodass die Verbindung zwischen Port und Vene gesichert ist.
    - Phase 5 Katheteranpassung: Der Katheter wird auf die richtige Länge zugeschnitten und an den Port angeschlossen.
    - Phase 6 Katheterkontrolle: Die Position des Katheters wird mittels Bildgebung kontrolliert, um eine korrekte Platzierung zu gewährleisten.
    - Phase 7 Abschluss: Die Wunde wird geschlossen, und der Eingriff wird abgeschlossen. Der Patient wird aus dem sterilen Bereich entlassen.

Die chirurgischen Schritte der laufenden Phase sind: {self.get_step_description(step_label)}
"""

        if step_label != 'Alltäglich':
            prompt += f"""
* reale Daten: Im Folgenden findest du ein Beispiele für reale Gespräche, die der Radiologe in der laufenden Phase geführt haben:
<reale Daten>
{sdg_helper.sample_real_phase(step_label)}</reale Daten>

* Satzgruppen: Häufig verwendete Ausdrücke der Radiologen bei realen Operationen wurden extrahiert und ähnliche Sätze der gleichen Phasen wurden gruppiert. Für die laufende Phase sind unten <{num2words(n_examples, lang='de')}> Beispielsätze mit Erklärungen angegeben:
    {sdg_helper.sample_pocap_example(phase=answer_df['Phase'].iloc[0], n_examples=n_examples)}

* Röntgenbildgebung: Wenn die Textspalte mit '*Atemkommando*' markiert ist, weise den Patienten an, seine Atmung während der Röntgenaufnahme zu kontrollieren. Beispielsätze, die du ähnliche Ausdrücke verwendest kann, sind folgendes: 
    -tief einatmen luft anhalten -ganz tief einatmen luft -einatmen luft anhalten atmen -bitte ganz tief einatmen -einatmen ausatmen luft anhalten -tief einathmen luft anhalten -nochmal einatmen ausatmen nochmal -nochmal tief einatmen luft -mal ganz tief einatmen -nochmal einatmen ausatmen luft -mal tief einatmen luft -luft anhalten atmen bewegen -nochmal ganz tief einatmen -ganz tief einathmen luft -bitte tief einatmen luft -bitte mal tief einatmen -einatmen ausatmen nochmal kräftig -drückt brennt bisschen haut -herr *PatientName* luft anhalten -bekommen gleich nochmal atemkommando>

* Fachwörter: Radiologen verwenden häufig die folgenden Fachbegriffe. Verwende ähnliche Fachbegriffe, wenn sie die laufende Tätigkeit und das Gespräch weitergeben. Die technischen Begriffe und ihre Bedeutungen sind wie folgt:
    - Kranial: in Richtung Kopf
    - Terumo, Terumonadel: präzise Injektionsnadel
    - Glomeruläre Filtrationsrate: Nieren-Blutfiltration pro Minute.
    - Decoderm: Creme zur Verhinderung von Infektionen
    - Omnistripes, Steri-Streifen: Wundverschlussstreifen
    - KM, Kontrastmittel: Substanz zur Bildgebung-Verstärkung
    - BV, Bildverstärkung: Verbesserung schwacher Bildsignale
    - DSA, DSA-Serie: Digitale Substraktionsangiographie
    - In-Stent-Stenose: Implantat zum Offenhalten von Arterien/Venen
    - Tamponade: Auffüllung von natürlich oder künstlichen Hohlräumen
    - Dialator: Werkzeug zum Erweitern einer Körperöffnung
    - Mecain, Mepivacain: Arzneimittel zur örtlichen Betäubung
    - Imeron: Kontrastmittel
    - Tegaderm: Durchsichtiger, selbstklebender Wundverband
    - Kavikula: Venenkanüle für intravenöse Zugänge
    - Vicryl: Resorbierbares chirurgisches Nahtmaterial
"""

        prompt += """
* Daten: Du erhältst einen Datensatz mit fehlenden Unterhaltungen im Abschnitt <Antwort>. Die Daten enthalten einen Index, die Startzeit der Rede, eine Angabe wer spricht, den gesprochenen Satz, Bezeichnungen für die laufende Operationsschritte und die Operationsphase.

* Aufgabe: Du wirst die Konversationen der Radiologen im Abschnitt <Antwort>, die mit '*Ausfüllen*' markiert sind, ergänzen, indem du die chirurgischen Phasen und Schritte berücksichtigst. Du ahmst die Persona des angegebenen Radiologen nach, wenn du Sätze erzeugst. Du wirst dann die Konversationen des gesamten Operation Teil für Teil erstellen. In diesem Teil wirst du die Daten für den angegebenen Abschnitt in der Vorlage <Antwort> generieren.

* Strategie: Zunächst fasst du deine Aufgabe in dem Abschnitt <Zusammenfassung> zusammen. Danach fahre mit der Generierung von Sätzen fort. Während du Sätze bildest: 1) Lies reale Gespräche im <reale Daten>-Abschnitt sorgfältig durch und verstehe, wie die Radiologen kommunizieren. 2) Lies frühere Gespräche, die im Punkt 'Kontext' unten gegeben sind, und analysiere, inwieweit Fortschritte bei diesem Thema gemacht worden sind. Nutze dafür die Informationen im Punkt „Überblick“ unten. 3) Plane das Tempo des Fortschritts und der dazugehörenden Gespräche entsprechend. Erzähle nicht alles zu Beginn der Phase/Schritt und wiederhole es viele Male. Plane sorgfältig und verteile die notwendigen Aktivitäten auf vorgegebene leere Gesprächsfelder. 4) Generiere Säzte. Verwende dein Verständnis und die Analyse der vorherigen drei Punkte bei der Satzbildung.

* Nörtralität: Bewahre in deinen Gesprächen möglichst einen neutralen ton. Erstelle eine gleichmäßige Anzahl von positiven und negativen Sätzen. Jeder positive Satz sollte durch einen ebenso negativen Satz ausgeglichen werden. Verwende neutrale Sprache, wo es möglich ist, aber achte darauf, dass die Anzahl positiver und negativer Sätze gleich bleibt.

* Sprache [SEHR WICHTIG!]:
    - Erzeuge natürliche Gespräche mit dem Patienten und dem Assistenten, und BEANTWORTE IHRE FRAGEN ODER KOMMENTARE.
    - MACHE GRAMMATIKALISCHE FEHLER. Echte Gespräche sind nicht streng strukturiert. Beobachte echte Daten und bilde ähnlich fehlerhafte, aber natürliche Sätze. 
    - Achte darauf, dass der Übergang zwischen den Phasen und Schritten natürlich erfolgt.
    - Halte deine Sätze in angemessener Länge. SCHREIBE IN JEDER TEXTFELD DURCHSCHNITTLICH <SIEBEN - ACHT WÖRTER>.
    - Um sicherzustellen, dass die generierten Sätze medizinisch korrekt sind, verwende die in den Spalten 'Schritt' und 'Phase' angegebenen Informationen und generiere korrekte Gespräche.
    - FOKUS LIEGT AUF PRÄZISEN MEDIZINISCHEN ANWEISUNGEN. Spreche präzise und direkt.
    - Wenn in der Spalte 'Schritt' 'Alltäglich' steht, führe ein alltägliches Gespräch. Fahre immer mit dem letzten Gesprächspunkt fort.
    - Betrachte die angegebenen SATZGRUPPEN UND N-GRAMME, ECHTE MENSCHLICHE SPRACHMUSTER ZU IMITIEREN.
    - IMPLIZIERE DEINE HANDLUNGEN MANCHMAL, anstatt sie immer ausdrücklich zu benennen.
    - Verwende manchmal typische deutsche Füllwörter, um natürlicher zu klingen.
    - Vermeide unnötige Floskeln.
    - Verwende keine Emojis.

* Format: Verwende die Vorlage im Abschnitt <Antwort> um deine Antwort zu geben und die Vorlage im Abschnitt <Zusammenfassung> um deine Zusammenfassung zu geben. Erzeuge nur die Vorlage mit der Überschrift in dem angegebenen <Antwort>-Abschnitt, füge keine neuen Zeilen hinzu. Gib deine anwort nur auf Deutsch und nutze CSV-Format im Abschnitt <Antwort> wie in der Vorlage. Verwende immer die Tags <Antwort> und </Antwort> am Anfang und Ende deiner Antwort, und <Zusammenfassung> und </Zusammenfassung> am Anfang und Ende deiner Zusammenfassung. Füge keine ``` codeblöcke oder ** Textblöcke hinzu, wenn sie nicht ausdrücklich dazu aufgefordert werden. Füge keine zusätzlichen Meldungen am Anfang oder Ende der Eingabeaufforderung ein.
"""
        if iteration:
            sentence_idx = step_df[(step_df['Schritt'] == step_label) &
                                   (step_df['Person'] == 'Radiologe')].shape[0]
            prompt += "\n* Kontext: Du ahmst die Persona der angegebenen Person in der Spalte 'Person' nach, wenn du Sätze erzeugst. Personen in der Spalte „Person“, die miteinander sprechen, berücksichtige bei der Erstellung neuer Sätze frühere Unterhaltungen. Bisherige Gespräche:\n"
            prompt += f"\n<Daten>\n{step_df.tail(n_context).to_csv(index=True, sep=';', index_label='Index')}</Daten>\n"
            if step_label != 'Alltäglich':
                prompt += f"\t * Überblick: Du wirst in mehreren Teilen der laufenden Operation über <{step_label}> insgesamt <{num2words(step_count, lang='de')}> Mal sprechen. Bisher <{num2words(sentence_idx+1, lang='de')}> Mal wurden gesprochen. Plane den weiteren Verlauf der Operation entsprechend. Im Abschnitt <Antwort> wirst du die gegebene Zeile erzeugen.\n"

        if self.problem != None:
            prompt += f"\n* Problem: Während einer Operation in einem Operationssaal können viele Dinge unerwartet passieren. Heute wirst du simulieren, dass im OP folgende Komplikation auftritt: <{self.problem}>. Reagiere bei einer geeigneten Gelegenheit darauf.\n"

        prompt += f"""
<Zusammenfassung>
Schreib hier
    1. Fasse die Persona des Radiologen zusammen, den du simulieren wirst.
    2. Was sind die chirurgische Schritte im angegebenen Datenbereich, die durchgeführt werden sollen?
    3. Welche Fachwörter passen zu den laufenden Schritten?
    4. Was ist die durchschnittliche Länge der Sätze, die du generieren wirst?
    5. Was sind die Sprachanweisungen, die als SEHR WICHTIG markiert sind und du folgen musst?
</Zusammenfassung>
"""
        if iteration:
            prompt += """\nIn diesem Teil wirst du mit der Generierung des nächsten Abschnitts der Operationsdaten fortfahren. Fülle nur die angegebenen Daten im Abschnitt <Antwort> aus. Berücksichtige deine vorherige Antworte um konsistente Konversionen zu generieren.\n"""
        
        prompt += f"\n<Antwort>\n{answer_df.to_csv(index=True, sep=';', index_label='Index')}</Antwort>\n"
        
        return prompt


    def get_assistant_prompt(self, iteration, step_df, answer_df, n_context=5):
        prompt = f"""{self.system_assistant}

Das Ziel ist es, realistische und einzigartige Gespräche in einem Operationssaal während einer Port-Katheter-Platzierung zu simulieren.

* Personal: Die Port-Katheter-Platzierung wird von einem Radiologen und einem medizinischen Assistenten in der radiologischen Abteilung durchgeführt. Der Radiologe ist verantwortlich für die Durchführung des Verfahrens, die Kommunikation mit dem Assistenten, um Anweisungen zu geben, und die Interaktion mit dem Patienten, um dessen Zustand zu überwachen und ihn ruhig zu halten. Der Assistent ist für die Vorbereitung steriler Materialien und die Bedienung des Röntgengeräts auf Anweisung des Radiologen zuständig. Der Patient ist die Person, die sich dem Verfahren unterzieht. Heute arbeitest du mit Radiologin/Radiolog <{self.radiologe}> und Patientin/Patient <{self.patient}>.

* Daten: Du erhältst einen Datensatz mit fehlenden Unterhaltungen im Abschnitt <Antwort>. Die Daten enthalten einen Index, die Startzeit der Rede, eine Angabe wer spricht, den gesprochenen Satz, Bezeichnungen für die laufende Operationsschritte und die Operationsphase.

* Aufgabe: Du wirst die Konversationen im Abschnitt <Antwort>, die mit '*Ausfüllen*' markiert sind, ergänzen. Die Splate 'Personen' zeigt, wer spricht gerade. Du berücksichtigst die bisherige Gespräche des Radiologen und Schritte und sprichst mit den Stil des vorgegebenen Personen. Schreibe in jeder 'Text' Spalte ungefähr <{num2words(10, lang='de')} Wörter>.
"""
        if iteration:
            prompt += "\n* Kontext: Du ahmst die Persona der angegebenen Person in der Spalte 'Person' nach, wenn du Sätze erzeugst. Personen in der Spalte „Person“, die miteinander sprechen, berücksichtige bei der Erstellung neuer Sätze frühere Unterhaltungen. Bisherige Gespräche:"
            prompt += f"\n<Daten>\n{step_df.tail(n_context).to_csv(index=True, sep=';', index_label='Index')}</Daten>\n"
            
        prompt += f"\n<Antwort>\n{answer_df.to_csv(index=True, sep=';', index_label='Index')}</Antwort>\n"

        return prompt


    def get_patient_prompt(self, iteration, step_df, answer_df, n_context=5):
        prompt = f"""{self.system_patient}

Das Ziel ist es, realistische und einzigartige Gespräche in einem Operationssaal während einer Port-Katheter-Platzierung zu simulieren.

* Personal: Die Port-Katheter-Platzierung wird von einem Radiologen und einem medizinischen Assistenten in der radiologischen Abteilung durchgeführt. Der Radiologe ist verantwortlich für die Durchführung des Verfahrens, die Kommunikation mit dem Assistenten, um Anweisungen zu geben, und die Interaktion mit dem Patienten, um dessen Zustand zu überwachen und ihn ruhig zu halten. Der Assistent ist für die Vorbereitung steriler Materialien und die Bedienung des Röntgengeräts auf Anweisung des Radiologen zuständig. Der Patient ist die Person, die sich dem Verfahren unterzieht. Heute arbeitest du mit Radiologin/Radiolog <{self.radiologe}> und Assistentin/Assistent <{self.assistent}>.

* Daten: Du erhältst einen Datensatz mit fehlenden Unterhaltungen im Abschnitt <Antwort>. Die Daten enthalten einen Index, die Startzeit der Rede, eine Angabe wer spricht, den gesprochenen Satz, Bezeichnungen für die laufende Operationsschritte und die Operationsphase.

* Aufgabe: Du wirst die Konversationen im Abschnitt <Antwort>, die mit '*Ausfüllen*' markiert sind, ergänzen. Die Splate 'Personen' zeigt, wer spricht gerade. Du berücksichtigst bisherige Gespräche und sprichst mit den Stil des vorgegebenen Personen. Schreibe in jeder 'Text' Spalte ungefähr <{num2words(10, lang='de')} Wörter>.
"""
        if iteration:
            prompt += "\n* Kontext: Du ahmst die Persona der angegebenen Person in der Spalte 'Person' nach, wenn du Sätze erzeugst. Personen in der Spalte „Person“, die miteinander sprechen, berücksichtige bei der Erstellung neuer Sätze frühere Unterhaltungen. Bisherige Gespräche:"
            prompt += f"\n<Daten>\n{step_df.tail(n_context).to_csv(index=True, sep=';', index_label='Index')}</Daten>\n"
            
        prompt += f"\n<Antwort>\n{answer_df.to_csv(index=True, sep=';', index_label='Index')}</Antwort>\n"

        return prompt


    def get_prompt(self, person, iteration, step_df, answer_df, step_label, step_count):
        if person == 'Radiologe':
            prompt = self.get_radiologe_prompt(iteration, step_df, answer_df, step_label, step_count)
        elif person == 'Assistent':
            prompt = self.get_assistant_prompt(iteration, step_df, answer_df)
        elif person == 'Patient':
            prompt = self.get_patient_prompt(iteration, step_df, answer_df)
        else:
            print('The person has a problem')
        return prompt