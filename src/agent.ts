import type { JobContext, JobProcess } from '@livekit/agents';
import {
  ServerOptions,
  cli,
  defineAgent,
  inference,
  llm,
  metrics,
  voice,
} from '@livekit/agents';

import * as deepgram from '@livekit/agents-plugin-deepgram';
import * as elevenlabs from '@livekit/agents-plugin-elevenlabs';
import * as livekit from '@livekit/agents-plugin-livekit';
import * as silero from '@livekit/agents-plugin-silero';
import { BackgroundVoiceCancellation } from '@livekit/noise-cancellation-node';
import dotenv from 'dotenv';
import { fileURLToPath } from 'node:url';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { z } from 'zod';

dotenv.config({ path: '.env.local' });

// --- DATABASE HELPERS ---
const DB_PATH = path.resolve(process.cwd(), 'clinic_data.json');

function loadClinicData(): any {
  if (!fs.existsSync(DB_PATH)) {
    console.error(`Database not found at ${DB_PATH}.`);
    return { clinics: [] };
  }
  const data = fs.readFileSync(DB_PATH, 'utf-8');
  return JSON.parse(data);
}

function saveClinicData(data: any) {
  fs.writeFileSync(DB_PATH, JSON.stringify(data, null, 4), 'utf-8');
}

// 10-Minute Slot Generator
function get10MinSlots(shift: any): string[] {
  const slots: string[] = [];
  if (!shift.startTime || !shift.endTime || shift.available === false || shift.status === 'booked') {
    return slots;
  }

  const parseTime = (timeStr: string) => {
    if (!timeStr || timeStr === "0:00" || timeStr === "00:00") return null;
    const [hours, minutes] = timeStr.split(':').map(Number);
    return new Date(2000, 0, 1, hours, minutes, 0, 0);
  };

  const formatTime = (d: Date) => {
    return d.toTimeString().substring(0, 5); // Returns "HH:MM"
  };

  const start = parseTime(shift.startTime);
  const end = parseTime(shift.endTime);
  const bStart = parseTime(shift.breakStart);
  const bEnd = parseTime(shift.breakEnd);

  if (!start || !end) return slots;

  let curr = start;
  while (curr.getTime() + 10 * 60000 <= end.getTime()) {
    let isBreak = false;
    if (bStart && bEnd) {
      if (curr.getTime() >= bStart.getTime() && curr.getTime() < bEnd.getTime()) {
        isBreak = true;
      }
    }

    let isBooked = false;
    if (shift.booked_appointments) {
      isBooked = shift.booked_appointments.some((b: any) => b.time === formatTime(curr));
    }

    if (!isBreak && !isBooked) {
      slots.push(formatTime(curr));
    }

    curr = new Date(curr.getTime() + 10 * 60000); // add 10 minutes
  }

  return slots;
}

const CITY_IMAGES: Record<string, string> = {
  amsterdam: "https://cdn.audleytravel.com/4861/3472/79/15985267-amsterdam-canal-in-the-autumn.jpg?w=800&q=80",
  paris: "https://res-4.cloudinary.com/gorealtravel/image/upload/,f_auto,q_50/v1563441342/production/marketing/city/5cf53801689bbf00089b7d1f/city_main_image/paris.webp",
  munich: "https://www.deutschland.de/sites/default/files/media/image/munich-germany-bavaria-tourism-travel-holidays-alps.jpg?w=800&q=80",
  vaasa: "https://storage.reveel.guide/photo/0199e714-481a-71f2-97a8-530c2e8154be/processed-0199e714-a83f-7197-960c-0d088dda2da0.webp"
};

class Health4TravelAgent extends voice.Agent {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private room: any;
  public userName: string;
  public agentName: string;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  constructor(metadata: string, room: any) {
    let clientName = 'Patient';
    try {
      const parsed = JSON.parse(metadata);
      if (parsed.clientName) clientName = parsed.clientName;
    } catch (e) {}

    const agentName = 'James';

    // Get real-time date logic so the LLM understands "Today"
    const today = new Date();
    const daysOfWeek = ['SUNDAY', 'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY'];
    const currentDayName = daysOfWeek[today.getDay()];
    const tomorrowDayName = daysOfWeek[(today.getDay() + 1) % 7];

    super({
      instructions: `# Persona & Tone
You are ${agentName}, the AI Medical Receptionist for Health4Travel. You are assisting ${clientName} with booking a doctor's appointment.
- Use a highly professional, polite, and reassuring tone.
- Use very short, simple sentences. One or two sentences maximum per turn.
- Speak in plain text. NEVER use markdown (no bold, no italics), lists, or symbols.
- Spell out numbers and times (e.g., "nine A M" instead of "9:00 AM").

# CRITICAL TIME CONTEXT
- Today is ${currentDayName}.
- Tomorrow is ${tomorrowDayName}.
- If the user says "today", you must pass "${currentDayName}" into your tools. 
- If the user says "tomorrow", you must pass "${tomorrowDayName}" into your tools.
- If they ask for "next week Monday", just pass "MONDAY".

# Conversational Flow
1. Greet the patient by name (${clientName}) and ask them which city they are traveling to.
2. When they mention a city (like Amsterdam, Paris, Vaasa), IMMEDIATELY call the 'showCityImage' tool silently.
3. PAY ATTENTION to the result of 'showCityImage':
   - If the tool says the city is fully booked, IMMEDIATELY tell the user: "I apologize, but all our slots in [City] are currently booked. Would you like to check another location?"
   - Only if the tool says the city has slots, ask them what day they would prefer.
4. IF they ask "What days are you open?" or "Show me available days", use the 'checkAvailableDays' tool.
5. Once they provide a specific day (or say today/tomorrow), use the 'checkAvailableSlots' tool.
6. Tell the user: "I have displayed the available appointment times on your screen. Please let me know which one you prefer, or just tap the button on your screen."
7. Once they choose a time, ask for their Phone Number to draft the booking.
8. Once you have their phone number, use the 'draftBooking' tool to show a pending ticket on their screen. **IMMEDIATELY ask the user: "Are you confirm booking on [Day] and [Time]?"**
9. IF the user says "Yes", "Confirm", or "Book it", use the 'confirmBooking' tool to save it to the database.
10. IF the user says "No", "Cancel", or "Cancel my booking", immediately use the 'cancelBooking' tool.

# Tool Usage Rules
- NEVER ask permission to show an image or slots. Just do it silently.
- Do not mention the names of your internal tools to the user.`,

      tools: {
        showCityImage: llm.tool({
          description: 'Show an image of the clinic location and immediately check if it has ANY available slots.',
          parameters: z.object({ city: z.string() }),
          execute: async ({ city }) => {
            const db = loadClinicData();
            const clinic = db.clinics.find((c: any) => c.city.toLowerCase() === city.toLowerCase());
            if (!clinic) return `No clinic found in ${city}.`;

            const imageUrl = CITY_IMAGES[city.toLowerCase()] || CITY_IMAGES['amsterdam'];
            const payload = JSON.stringify({ type: 'show_image', url: imageUrl, title: clinic.clinic_name });

            if (this.room) await this.room.localParticipant.publishData(new TextEncoder().encode(payload), { reliable: true });

            // Check if there are ANY available slots in this city
            const hasAvailableSlots = clinic.slots.some((s: any) => s.available && s.status !== 'booked');

            if (!hasAvailableSlots) {
              return `Image shown. WARNING: There are NO available slots in ${city}. All slots are booked out. Tell the user immediately that booking is not available in ${city} and ask if they want to check another city.`;
            }

            return `Image of ${city} clinic shown on screen. The clinic has availability. Ask the user what day they would prefer.`;
          }
        }),

        checkAvailableDays: llm.tool({
          description: 'Check which DAYS the clinic is open if the user asks for available days.',
          parameters: z.object({ city: z.string() }),
          execute: async ({ city }) => {
            const db = loadClinicData();
            const clinic = db.clinics.find((c: any) => c.city.toLowerCase() === city.toLowerCase());
            if (!clinic) return `No slots found in ${city}.`;

            const availableDays = [...new Set(clinic.slots.filter((s: any) => s.available && s.status !== 'booked').map((s: any) => s.day))];
            
            if (availableDays.length === 0) return `I'm sorry, we are completely booked in ${city}.`;
            return `We have appointments available on the following days: ${availableDays.join(', ')}. Please ask the user which of these days they prefer.`;
          }
        }),

        checkAvailableSlots: llm.tool({
          description: 'Check available 10-minute appointment slots for a specific day and push buttons to the user screen.',
          parameters: z.object({
            city: z.string().describe('The city name'),
            day: z.string().describe('The day of the week (e.g., MONDAY). DO NOT PASS "TODAY", pass the actual name of the day.'),
          }),
          execute: async ({ city, day }) => {
            const db = loadClinicData();
            const clinic = db.clinics.find((c: any) => c.city.toLowerCase() === city.toLowerCase());
            if (!clinic) return `No slots found in ${city}.`;

            const targetDay = day.toUpperCase();
            const availableShifts = clinic.slots.filter((s: any) => s.day === targetDay && s.available && s.status !== 'booked');

            if (availableShifts.length === 0) {
              return `I'm sorry, we don't have any available slots on ${targetDay} in ${city}. Please ask them for another day.`;
            }

            const allGeneratedSlots: any[] = [];
            for (const shift of availableShifts) {
              const times = get10MinSlots(shift);
              for (const t of times) {
                allGeneratedSlots.push({ start_time: t, day: shift.day, clinic_name: clinic.clinic_name, slot_id: shift.slot_id });
              }
            }

            if (allGeneratedSlots.length === 0) {
              return `I'm sorry, all slots for ${targetDay} are currently booked out.`;
            }

            const payload = JSON.stringify({ type: 'show_slots', slots: allGeneratedSlots });
            if (this.room) await this.room.localParticipant.publishData(new TextEncoder().encode(payload), { reliable: true });

            return `Found ${allGeneratedSlots.length} available slots. The buttons are now visible on the screen. Ask them to select a specific time.`;
          }
        }),

        draftBooking: llm.tool({
          description: 'Draft the appointment and show a pending ticket for the user to confirm BEFORE saving it to the database. Use this after getting the phone number.',
          parameters: z.object({
            city: z.string(),
            day: z.string(),
            time: z.string(),
            phone_number: z.string(),
          }),
          execute: async ({ city, day, time, phone_number }) => {
            const db = loadClinicData();
            let targetClinic = db.clinics.find((c: any) => c.city.toLowerCase() === city.toLowerCase());
            if (!targetClinic) return `Booking failed. Invalid city.`;

            // Close the slots side panel so it doesn't clutter the screen
            const closeSlotsPayload = JSON.stringify({ type: 'close_slots' });

            const ticket = {
              status: "PENDING CONFIRMATION",
              patient_name: clientName, 
              phone_number: phone_number, 
              day: day.toUpperCase(), 
              time: time,
              clinic_name: targetClinic.clinic_name, 
              address: targetClinic.streetAddress || targetClinic.address
            };

            const payload = JSON.stringify({ type: 'show_ticket', ticket: ticket });
            
            if (this.room) {
              await this.room.localParticipant.publishData(new TextEncoder().encode(closeSlotsPayload), { reliable: true });
              await this.room.localParticipant.publishData(new TextEncoder().encode(payload), { reliable: true });
            }

            return `Pending ticket shown on screen. You MUST ask the user exactly: "Are you confirm booking on ${day} and ${time}?"`;
          }
        }),

        confirmBooking: llm.tool({
          description: 'Finalize and save the booking ONLY AFTER the user explicitly says YES or CONFIRM to the draft.',
          parameters: z.object({
            city: z.string(),
            day: z.string(),
            time: z.string(),
            phone_number: z.string(),
          }),
          execute: async ({ city, day, time, phone_number }) => {
            const db = loadClinicData();
            let targetClinic = null; 
            let targetShift = null;

            for (const c of db.clinics) {
              if (c.city.toLowerCase() === city.toLowerCase()) {
                for (const s of c.slots) {
                  if (s.day === day.toUpperCase()) { 
                    targetClinic = c; 
                    targetShift = s; 
                    break; 
                  }
                }
              }
            }

            if (!targetClinic || !targetShift) return `Confirmation failed. Invalid city or day.`;
            if (!targetShift.booked_appointments) targetShift.booked_appointments = [];
            
            if (targetShift.booked_appointments.some((b:any) => b.time === time)) {
              return `I'm sorry, that specific time slot was just taken. Pick another time.`;
            }

            // Save to database
            targetShift.booked_appointments.push({ time: time, patient_name: clientName, phone_number: phone_number });
            saveClinicData(db);

            const ticket = {
              status: "CONFIRMED ✅",
              patient_name: clientName, 
              phone_number: phone_number, 
              day: day.toUpperCase(), 
              time: time,
              clinic_name: targetClinic.clinic_name, 
              address: targetClinic.streetAddress || targetClinic.address
            };

            const payload = JSON.stringify({ type: 'show_ticket', ticket: ticket });
            const closeImgPayload = JSON.stringify({ type: 'close_image' });

            if (this.room) {
              await this.room.localParticipant.publishData(new TextEncoder().encode(closeImgPayload), { reliable: true });
              await this.room.localParticipant.publishData(new TextEncoder().encode(payload), { reliable: true });
            }

            return `Successfully saved to database. Tell the user their ticket is confirmed and they are all set!`;
          }
        }),

        cancelBooking: llm.tool({
          description: 'Cancel an appointment from the database if the user requests it.',
          parameters: z.object({
            city: z.string(),
            day: z.string(),
            time: z.string(),
          }),
          execute: async ({ city, day, time }) => {
            const db = loadClinicData();
            let foundAndDeleted = false;

            for (const c of db.clinics) {
              if (c.city.toLowerCase() === city.toLowerCase()) {
                for (const s of c.slots) {
                  if (s.day === day.toUpperCase() && s.booked_appointments) {
                    const initialLength = s.booked_appointments.length;
                    s.booked_appointments = s.booked_appointments.filter((b:any) => b.time !== time);
                    if (s.booked_appointments.length < initialLength) {
                      foundAndDeleted = true;
                    }
                  }
                }
              }
            }

            // Always clear the ticket from the UI, regardless of DB presence 
            // (e.g., if they decline the draft confirmation)
            const closeTicketPayload = JSON.stringify({ type: 'show_ticket', ticket: null });
            if (this.room) await this.room.localParticipant.publishData(new TextEncoder().encode(closeTicketPayload), { reliable: true });

            if (foundAndDeleted) {
              saveClinicData(db);
              return `The appointment for ${day} at ${time} has been successfully cancelled from the database. Inform the user.`;
            } else {
              return `The draft booking has been cancelled and cleared from the screen.`;
            }
          }
        }),
      }
    });

    this.room = room;
    this.userName = clientName;
    this.agentName = agentName;
  }
}

export default defineAgent({
  prewarm: async (proc: JobProcess) => {
    proc.userData.vad = await silero.VAD.load();
  },
  entry: async (ctx: JobContext) => {
    try {
      await ctx.connect();
      console.log('✅ Connected to LiveKit Room');

      const stt = new deepgram.STT({ apiKey: process.env.DEEPGRAM_API_KEY!, profanityFilter: true });
      const llm_model = new inference.LLM({ model: 'openai/gpt-4.1-mini' });
      const tts = new elevenlabs.TTS({
        apiKey: process.env.ELEVEN_API_KEY!, enableLogging: true, voiceId: process.env.ELEVEN_VOICE_ID!, language: 'en', model: 'eleven_flash_v2_5'
      });

      const session = new voice.AgentSession({
        stt: stt, llm: llm_model, tts: tts, turnDetection: new livekit.turnDetector.MultilingualModel(), vad: ctx.proc.userData.vad! as silero.VAD,
        voiceOptions: { preemptiveGeneration: true, allowInterruptions: true, minInterruptionDuration: 1.2, minInterruptionWords: 5, minEndpointingDelay: 0.6, maxEndpointingDelay: 3.0, maxToolSteps: 10 },
      });

      let participantMetadata = '{}';
      const remoteParticipants = Array.from(ctx.room.remoteParticipants.values());
      if (remoteParticipants.length > 0) {
        if (remoteParticipants[0]?.metadata) participantMetadata = remoteParticipants[0].metadata;
      }

      const agentInstance = new Health4TravelAgent(participantMetadata, ctx.room);

      const usageCollector = new metrics.UsageCollector();
      session.on(voice.AgentSessionEventTypes.MetricsCollected, (ev) => { metrics.logMetrics(ev.metrics); usageCollector.collect(ev.metrics); });
      ctx.addShutdownCallback(async () => { console.log(`Usage: ${JSON.stringify(usageCollector.getSummary())}`); });

      await session.start({ agent: agentInstance, room: ctx.room, inputOptions: { noiseCancellation: BackgroundVoiceCancellation() }});

      console.log('✅ AI Session Started successfully');
      await new Promise((resolve) => setTimeout(resolve, 1000));

      await session.say(
        `Hello ${agentInstance.userName}! Welcome to Health 4 Travel. I am ${agentInstance.agentName}, your Smart Clinic Assistant. Which city are you looking to book a doctor in today?`
      );

    } catch (error) {
      console.error('❌ ERROR IN AGENT ENTRY:', error);
    }
  },
});

cli.runApp(new ServerOptions({ agent: fileURLToPath(import.meta.url) }));