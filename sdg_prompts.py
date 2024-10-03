import sdg_helper
from num2words import num2words


class SDGPrompts:
    def __init__(self):
        pass

    def init_prompts(self):
        self.system_radiologe = f"Du bist ein hilfsbereiter Assistent, der realistische Gespräche führt, indem er eine bestimmte Persona simuliert. Die Persona gehört zu einem Radiologen, der bei einem Krankenhaus in Deutschland arbeitet. Du simulierst {self.radiologe}"
        self.system_assistant = f"Du bist hilfsbereiter Assistent, der realistische Gespräche führt, indem er eine bestimmte Persona simuliert. Die Persona gehört zu einem medizinischen Assistenten, der bei einem Krankenhaus in Deutschland arbeitet. Du simulierst {self.assistent}"
        self.system_patient = f"Du bist ein hilfreicher Assistent, der realistische Gespräche führt, indem er eine bestimmte Persona simuliert. Die Persona gehört zu einem Patient, derin einem Krankenhaus in Deutschland operiert wird. Du simulierst {self.patient}"

        self.base_prompt = f"""Das Ziel ist es, realistische und einzigartige Gespräche in einem Operationssaal während einer Port-Katheter-Platzierung zu simulieren.

* Personal: Die Port-Katheter-Platzierung wird von einem Radiologen und einem medizinischen Assistenten in der radiologischen Abteilung durchgeführt. Der Radiologe ist verantwortlich für die Durchführung des Verfahrens, die Kommunikation mit dem Assistenten, um Anweisungen zu geben, und die Interaktion mit dem Patienten, um dessen Zustand zu überwachen und ihn ruhig zu halten. Der Assistent ist für die Vorbereitung steriler Materialien und die Bedienung des Röntgengeräts auf Anweisung des Radiologen zuständig. Der Patient ist die Person, die sich dem Verfahren unterzieht. 

* Operation: Chirurgische Phasen und chirurgische Schritte darstellen eine typische Operation. Die Phasen beziehen sich auf die großen Abschnitte des Verfahrens, in denen die wichtigsten Schritte beschrieben werden. Chirurgische Schritte sind die spezifischen Aufgaben, die innerhalb jeder Phase ausgeführt werden sollen. Operationen folgen im Allgemeinen dieser Reihenfolge der Ereignisse. Die Phasen und Schritte der Port-Katheter-Platzierung Operation sind folgendes:
    - Phase 0: Vorbereitung. Schritt 0.1 Positionierung des Patienten auf dem Tisch: Der Patient wird in eine stabile, komfortable Position gebracht, meist in Rückenlage. Dies ist wichtig für den Zugang zu den Venen und die Sicherheit während der Operation.
    - Phase 0: Vorbereitung. Schritt 0.2 Tisch fährt hoch: Der Operationstisch wird auf eine ergonomische Höhe für den Chirurgen und Radiologen gebracht, um den Eingriff effizient durchzuführen.
    - Phase 0: Vorbereitung. Schritt 0.3 Radiologe (Chirurg) sterilisiert sich: Der Radiologe, der die Operation übernimmt, bereitet sich durch Sterilisation und Anziehen steriler Kleidung vor, um Infektionen zu verhindern.
    - Phase 0: Vorbereitung. Schritt 0.4 Vorbereitung des sterilen Materials: Alle benötigten Materialien (Katheter, Nadeln, Nahtmaterial, etc.) werden in einem sterilen Feld vorbereitet, um eine Kontamination zu vermeiden.
    - Phase 0: Vorbereitung. Schritt 0.5 Patient steril abgedeckt: Der Patient wird steril abgedeckt, damit nur der operative Bereich freiliegt. Dies reduziert das Risiko von Infektionen.
    - Phase 1: Punktion. Schritt 1.1 Lokale Anästhesie: Das operative Gebiet wird lokal betäubt, um den Patienten während des Eingriffs schmerzfrei zu halten.
    - Phase 1: Punktion. Schritt 1.2 Ultraschallgeführte Punktion: Mit Ultraschall wird die Zielvene (meist die Vena subclavia oder die Vena jugularis interna) angesteuert und punktiert. Diese Methode verbessert die Sicherheit und Präzision, da der Zugang zur Vene sichtbar kontrolliert wird.
    - Phase 2: Führungsdraht. Schritt 2.1 Röntgenmaschine fährt ein: Die Röntgenmaschine wird in Position gebracht, um die nachfolgenden Schritte bildgebend zu überwachen.
    - Phase 2: Führungsdraht. Schritt 2.2 Durchleuchtung im Bereich der Subklavia: Eine Durchleuchtung stellt sicher, dass der Führungsdraht korrekt in die Vene eingeführt wird.
    - Phase 2: Führungsdraht. Schritt 2.3 Durchleuchtung im Bereich der Vena cava inferior (VCI): Der Führungsdraht wird bis zur Vena cava inferior vorgeschoben, um die spätere Position des Katheters zu bestätigen.
    - Phase 2: Führungsdraht. Schritt 2.4 Röntgenmaschine fährt heraus: Nach erfolgreicher Platzierung des Drahtes wird die Röntgenmaschine für den nächsten Schritt zurückgezogen.
    - Phase 3: Pouchvorbereitung-und-Katheterplatzierung. Schritt 3.1 Lokale Anästhesie: Das Gebiet über dem Schlüsselbein wird lokal betäubt, um den Schnitt für die Portkammer vorzubereiten.
    - Phase 3: Pouchvorbereitung-und-Katheterplatzierung. Schritt 3.2 Inzision: Es wird ein Hautschnitt durchgeführt, um Zugang zum Unterhautgewebe zu schaffen, wo der Port später platziert wird.
    - Phase 3: Pouchvorbereitung-und-Katheterplatzierung. Schritt 3.3 Pouch-Vorbereitung: Mit einem stumpfen Präparierinstrument wird eine kleine Tasche im Gewebe geschaffen, um Platz für die Portkammer zu machen.
    - Phase 3: Pouchvorbereitung-und-Katheterplatzierung. Schritt 3.4 Hülleplatzierung: Eine Hülse (Schleuse) wird um den Führungsdraht gelegt, um den Katheter über den Draht in die Vene einzuführen.
    - Phase 4: Katheterpositionierung. Shritt 4.1 Röntgenmaschine fährt ein: Die Röntgenmaschine wird erneut aktiviert, um die Platzierung des Katheters zu überwachen.
    - Phase 4: Katheterpositionierung. Shritt 4.2 Durchleuchtung des VCI-Bereichs: Die Position des Katheters in der Vena cava inferior wird überprüft, um sicherzustellen, dass er korrekt platziert ist.
    - Phase 4: Katheterpositionierung. Shritt 4.3 Positionierung des Katheters: Der Katheter wird bis in die richtige Tiefe vorgeschoben, meist knapp oberhalb der rechten Herzvorhofgrenze.
    - Phase 5: Katheteranpassung. Schritt 5.1 Kürzen des Katheters: Der Katheter wird auf die richtige Länge gekürzt, um eine optimale Funktion und Platzierung zu gewährleisten.
    - Phase 5: Katheteranpassung. Schritt 5.2 Röntgenmaschine fährt aus: Die Bildgebung wird abgeschlossen, da die endgültige Katheterplatzierung gesichert ist.
    - Phase 5: Katheteranpassung. Schritt 5.3 Anschluss des Katheters an die Portkapsel: Der Katheter wird mit der Portkapsel verbunden, die das Medikament später in den Blutkreislauf leitet.
    - Phase 5: Katheteranpassung. Schritt 5.4 Positionierung der Portkapsel im Pouch: Die Portkapsel wird in den zuvor geschaffenen Pouch implantiert und fixiert.
    - Phase 5: Katheteranpassung. Schritt 5.5 Chirurgische Naht: Der Hautschnitt wird in mehreren Schichten vernäht, um die Implantationsstelle zu verschließen.
    - Phase 5: Katheteranpassung. Schritt 5.6 Punktion der Portkapsel: Zum Testen der Portfunktion wird die Kapsel punktiert, um sicherzustellen, dass alles korrekt verbunden ist.
    - Phase 6: Katheterkontrolle. Schritt 6.1 Röntgenmaschine fährt ein: Die Röntgenmaschine wird aktiviert, um die Funktion des Katheters zu überprüfen.
    - Phase 6: Katheterkontrolle. Schritt 6.2 Digitale Subtraktionsangiographie des Brustbereichs: Eine Kontrastmittelgabe mit digitaler Subtraktionsangiographie stellt sicher, dass der Katheter durchgängig und richtig platziert ist.
    - Phase 6: Katheterkontrolle. Schritt 6.3 Röntgenmaschine fährt in Parkposition aus: Nach der abschließenden Überprüfung wird die Röntgenmaschine endgültig deaktiviert.
    - Phase 7: Abschluss. Schritt 7.1 Steriles Pflaster auflegen: Über der Naht wird ein steriles Pflaster angebracht, um die Wunde zu schützen.
    - Phase 7: Abschluss. Schritt 7.2 Tisch fährt nach unten: Der Operationstisch wird abgesenkt, um den Patienten sicher vom Tisch zu transferieren.

* Satzgruppen: Häufig verwendete Ausdrücke der Radiologen bei realen Operationen wurden extrahiert und gruppiert. Für jede chirurgische Phase werden im Folgenden Satzgruppen mit drei Beispielsätzen und der Anzahl des Vorkommens im Datensatz angegeben:
    - In der Phase Vorbereitung: fünf Mal Aussagen wie 1) 'Nicht hinlangen, keine Angst, ich mache es gleich so, dass Sie wieder rausschauen.' 2) 'Ich decke Sie mal ein bisschen zu, aber ich mache es sofort wieder weg.' 3) 'Ich gehe mal kurz über die Augen, deswegen bitte kurz die Augen schließen.' wurden vor der Tuchabdeckung geäußert, um den Patienten zu beruhigen. Fünf Mal Aussagen wie 1) 'Nehmen Sie mal das sterile Tuch.' 2) 'Ich decke Sie gleich mit einem OP-Tuch ab.' 3) 'Wir legen jetzt schon mal ein steriles Tuch bei Ihnen auf.' wurden geäußert, um den Patienten über den bevorstehenden Schritt „0.5) Patient steril abgedeckt“ zu informieren.
    - In der Phase Punktion: 35 Mal Aussagen ähnlich wie 1) "Bitte pressen Sie kräftig in den Bauch, als ob Sie auf die Toilette müssten.", 2) "Nochmal kräftig in den Bauch reinpressen, bitte." 3)"Einatmen, ausatmen und dann kräftig in den Bauch pressen." und sieben Mal Aussagen ähnlich wie 1) "Bitte atmen Sie tief ein." 2)"Halten Sie die Luft an." 3) "Atmen Sie langsam weiter." wurden während des chirurgischen Schritts '1.2 Ultraschallgeführte Punktion' geäußert, um die Punktionsstelle besser sichtbar zu machen.
    - In der Phase Führungsdraht: sieben Mal Aussagen ähnlich wie 1) "Bitte blenden Sie oben und unten auf." 2) "Blenden Sie links und rechts ein." 3) "Blenden Sie oben, unten, links und rechts ein." vor oder während der Röntgenaufnahme geäußert, um den Kollimator mithilfe der Assistentin zu steuern. Vier Mal Aussagen ähnlich wie 1) "Wir haben den schwierigen Schritt geschafft und sind in die Vene gekommen." 2) "Wir sind bereits in der Vene, das hat super geklappt." 3) "Den schwierigen Schritt haben wir geschafft, jetzt wird es noch ein bisschen pieksen." wurden geäußert, um den Patienten darüber zu informieren, dass die Phase 'Führungsdraht' zu Ende gekommen ist.
    - In der Phase Pouchvorbereitung-und-Katheterplatzierung: 23 Mal Aussagen wie 1) "Jetzt wird die lokale Betäubung verabreicht." 2) "Wir warten, bis die Betäubung wirkt." 3) "Die Betäubung wirkt gut bei Ihnen." wurden geäußert, um den Patienten über den Schritt „3.1) Lokale Anästhesie“ zu informieren.
    - In der Phase Katheterpositionierung: fünf Mal Aussagen wie 1) "Bitte atmen Sie tief ein." 2) "Atmen Sie ganz tief ein und halten Sie die Luft an." 3) "Jetzt bitte tief einatmen." wurden während der Röntgenaufnahme geäußert, um die Durchleuchtungsstelle leichter erkennbar zu machen. Vier Mal Aussagen wie 1) "Der Portschlauch ist jetzt platziert, nun verbinden wir ihn mit der Katheter. 2) "Wir verbinden den Schlauch mit der Kammer." 3) "Wir stellen jetzt die Länge des Portschlauchs ein." wurden geäußert, um den Patient über den Schritt '4.3) Positionierung des Katheters' zu informieren. Drei Mal Aussagen wie 1) "Wir warten kurz, bis die Blutstillung abgeschlossen ist." 2) "Wir lassen es kurz ruhen, damit es nicht blutet." 3) "Wir warten, bis die Tasche hier bis ins Blut raufkönnt." geäußert.
    - In der Phase Katheteranpassung: 43 Mal Aussagen wie 1) "Bitte atmen Sie tief ein und halten Sie die Luft an." 2) "Atmen Sie ganz tief aus und pressen Sie alles raus." 3) "Atmen Sie ruhig und halten Sie die Luft an, wenn ich es sage." wurden während der Röntgenaufnahme geäußert, um die Durchleuchtungsstelle leichter erkennbar zu machen.
    - In der Phase Katheterkontrolle 21 Mal Aussagen wie 1) "Bitte atmen Sie tief ein und halten Sie die Luft an." 2) "Halten Sie die Luft an und bewegen Sie sich nicht." 3) "Sie bekommen gleich ein Atemkommando, bitte atmen Sie tief ein und halten Sie die Luft an." wurden während der Röntgenaufnahme geäußert, um die Durchleuchtungsstelle leichter erkennbar zu machen.
    - In der Phase Abschluss: neun Mal Aussagen wie 1) "Wir kleben jetzt ein Pflaster darauf." 2) "Das Pflaster kann später einfach abgezogen werden." 3) "Sie bekommen ein Ersatzpflaster mit." wurden geäußert, um den Patient über den Schritt '7.1) Steriles Pflaster auflegen' zu informieren.
    
    Achte darauf, ähnliche Ausdrücke zu verwenden, wenn sie bei der Bildung von Sätzen natürlich in den Kontext passen. Bevor du sie verwendest, solltest du jedoch bedenken, wie oft diese Ausdrücke normalerweise im Datensatz vorkommen. Dies wird dazu beitragen, dass der Text realistischer wirkt.

* Daten: Du erhältst einen Datensatz mit fehlenden Unterhaltungen im Abschnitt <Antwort>. Die Daten enthalten einen Index, die Startzeit der Rede, eine Angabe wer spricht, den gesprochenen Satz, Bezeichnungen für die laufende Operationsschritte und die Operationsphase.

* Aufgabe: Du wirst die Konversationen im Abschnitt <Antwort>, die mit '*Ausfüllen*' markiert sind, ergänzen. Die Splate 'Personen' zeigt, wer spricht gerade. Du berücksichtigst die chirurgischen Phasen und Schritte und sprichst mit den Stil des vorgegebenen Personen.

* Strategie: Zunächst fasst du deine Aufgabe in dem Abschnitt <Zusammenfassung> zusammen. Du wirst dann die Konversationen des gesamten Operation Teil für Teil erstellen. In diesem Teil wirst du die Daten für den angegebenen Abschnitt in der Vorlage <Antwort> generieren. Um sicherzustellen, dass die generierten Sätze medizinisch korrekt sind, verwende die in den Spalten 'Schritt' und 'Phase' angegebenen Informationen und generiere geeignete Gespräche. Betrachte auch die angegebenen Satzgruppen, um sich inspirieren zu lassen. Wenn in der Spalte 'Schritt' bereits 'Alltäglich' steht, erstelle stattdessen einen themenfremden Satz, um ein alltägliches Gespräch mit den anderen Personen zu beginnen. Ein Themavorschlag ist: {self.topic}.

* Format: Verwende die Vorlage im Abschnitt <Antwort> um deine Antwort zu geben und die Vorlage im Abschnitt <Zusammenfassung> um deine Zusammenfassung zu geben. Gib deine anwort nur auf Deutsch und nutze CSV-Format im Abschnitt <Antwort> wie in der Vorlage. Verwende immer die Tags <Antwort> und </Antwort> am Anfang und Ende deiner Antwort, und <Zusammenfassung> und </Zusammenfassung> am Anfang und Ende deiner Zusammenfassung.
"""
        self.data_prompt = "\n* Kontext: Du ahmst die Persona der angegebenen Person in der Spalte 'Person' nach, wenn du Sätze erzeugst. Personen in der Spalte „Person“, die miteinander sprechen, berücksichtige bei der Erstellung neuer Sätze frühere Unterhaltungen. Bisherige Gespräche:"

        self.iteration_prompt = """\nIn diesem Teil wirst du mit der Generierung des nächsten Abschnitts der Operationsdaten fortfahren. Fülle die angegebenen Daten im Abschnitt <Antwort> aus. Berücksichtige deine vorherige Antworte um konsistente Konversionen zu generieren."""

        self.summary_prompt = """
<Zusammenfassung>
Schreib hier
    1. den Namen der Operation
    2. Fasse den Sprachstil zusammen, den du simulieren wirst.
    3. chirurgische Schritte im angegebenen Datenbereich, die durchgeführt werden sollen
    4. Welche Satzgruppen könnten relevant sein?
</Zusammenfassung>
"""

    def init_OR(self):
        self.radiologe = sdg_helper.sample_radiologe()
        self.assistent = sdg_helper.sample_assistent()
        self.patient = sdg_helper.sample_patient()
        self.topic = sdg_helper.sample_daily_topic()
        self.init_prompts()

    def get_step_remainder_prompt(self, person, step_label, step_count, idx):
        return f"""\n* Überblick: Im angegebenen Datenteil siehst du einen Ausschnitt aus dem Gesamtdatensatz. Aber, insgesamt wird die Person <{person}> während des aktuellen Operationsschritts <{step_label}> <{num2words(step_count, lang='de')}> Mal sprechen. In diesem Teil beginnst du mit dem Satzindex <{num2words(idx+1, lang='de')}>. Plane deine Sätze entsprechend.\n"""

    def get_initial_prompt(self, df, person, step_label, step_count):
        prompt = self.base_prompt
        if step_label != 'Alltäglich':
            prompt += self.get_step_remainder_prompt(
                person, step_label, step_count, 0)
        prompt += self.summary_prompt
        prompt += f"\n<Antwort>\n{df.to_csv(index=True, sep=';', index_label='Index')}</Antwort>\n"
        return prompt

    def get_iteration_prompt(self, step_df, sub_df, person, step_label, step_count, context_length=25):
        prompt = self.base_prompt
        if step_label != 'Alltäglich':
            sentence_idx = step_df[(step_df['Schritt'] == step_label) &
                                   (step_df['Person'] == person)].shape[0]
            prompt += self.get_step_remainder_prompt(
                person, step_label, step_count, sentence_idx)
        prompt += self.data_prompt
        prompt += f"\n<Daten>\n{step_df.tail(context_length).to_csv(index=True, sep=';', index_label='Index')}</Daten>\n"
        prompt += self.iteration_prompt
        prompt += f"\n<Antwort>\n{sub_df.to_csv(index=True, sep=';', index_label='Index')}</Antwort>\n"
        return prompt
