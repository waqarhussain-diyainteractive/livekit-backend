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

// --- TIME HELPER (Universal Formatting) ---
function normalizeTime(timeStr: string): string | null {
  if (!timeStr || timeStr === "0:00" || timeStr === "00:00") return null;
  const cleanStr = timeStr.toLowerCase().replace(/\s/g, ''); 
  
  let hours = 0;
  let minutes = 0;
  
  const isPM = cleanStr.includes('pm');
  const isAM = cleanStr.includes('am');
  const timeOnly = cleanStr.replace('am', '').replace('pm', '');
  
  if (timeOnly.includes(':')) {
    const parts = timeOnly.split(':');
    hours = parseInt(parts[0] || '0', 10);
    minutes = parseInt(parts[1] || '0', 10) || 0;
  } else {
    hours = parseInt(timeOnly || '0', 10);
  }
  
  if (isNaN(hours)) return null;
  if (isPM && hours < 12) hours += 12;
  if (isAM && hours === 12) hours = 0;
  
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
}

// --- SLOT GENERATOR ---
function get10MinSlots(shift: any): string[] {
  const slots: string[] = [];
  if (!shift.startTime || !shift.endTime || shift.available === false || shift.status === 'booked') {
    return slots;
  }

  const startStr = normalizeTime(shift.startTime);
  const endStr = normalizeTime(shift.endTime);
  const bStartStr = normalizeTime(shift.breakStart);
  const bEndStr = normalizeTime(shift.breakEnd);

  if (!startStr || !endStr) return slots;

  const toDate = (hhmm: string) => {
    const [h, m] = hhmm.split(':').map(Number);
    return new Date(2000, 0, 1, h, m);
  };

  const start = toDate(startStr);
  const end = toDate(endStr);
  const bStart = bStartStr ? toDate(bStartStr) : null;
  const bEnd = bEndStr ? toDate(bEndStr) : null;

  const formatTime = (d: Date) => d.toTimeString().substring(0, 5); // Returns "HH:MM"

  let curr = start;
  while (curr.getTime() + 10 * 60000 <= end.getTime()) {
    let isBreak = false;
    if (bStart && bEnd && curr.getTime() >= bStart.getTime() && curr.getTime() < bEnd.getTime()) {
      isBreak = true;
    }

    let isBooked = false;
    if (shift.booked_appointments) {
      const currFormatted = formatTime(curr);
      isBooked = shift.booked_appointments.some((b: any) => {
        const normalizedDbTime = normalizeTime(b.time);
        return normalizedDbTime === currFormatted;
      });
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
  public agentName: string;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  constructor(room: any) {
    const agentName = 'Maria';

    // Get real-time date logic so the LLM understands "Today"
    const today = new Date();
    const daysOfWeek = ['SUNDAY', 'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY'];
    const currentDayName = daysOfWeek[today.getDay()];
    const tomorrowDayName = daysOfWeek[(today.getDay() + 1) % 7];

    super({
      instructions: `# Persona & Tone
You are ${agentName}, a premium Global Medical Concierge for Health4Travel. You assist international patients with booking doctor appointments seamlessly.
- Tone: Highly professional, warm, reassuring, and culturally accommodating.
- Language: Speak in clear, simple, and universally understood English. Keep sentences very short. One or two sentences maximum per turn.
- Spoken Times: Always speak times naturally (e.g., say "two thirty P M" instead of "14:30"). 
- Text: Speak in plain text ONLY. No markdown, no symbols, no lists.
- CRITICAL: NEVER read long lists of available time slots. It sounds robotic and overwhelming over the phone.

# CRITICAL TIME CONTEXT
- Today is ${currentDayName}.
- Tomorrow is ${tomorrowDayName}.
- If the user says "today", pass "${currentDayName}" into your tools. 
- If the user says "tomorrow", pass "${tomorrowDayName}" into your tools.

# Conversational Flow (STEP BY STEP)
1. Greet the patient and ask them which city they are traveling to.
2. When they mention a city, IMMEDIATELY call 'showCityImage' silently.
   - If booked out: Apologize warmly and ask if they want to check another city.
   - If slots exist: Ask them what day they would prefer.
3. IF they ask "What days are you open?", use the 'checkAvailableDays' tool.
4. Once they provide a specific day, use the 'checkAvailableSlots' tool.
   - DO NOT read the array of available slots!
   - Look at the tool data. If there is a break time, say: "The clinic is open from [Open Time] to [Close Time], with a break from [Break Start] to [Break End]. Appointments are 10 minutes long. What time works best for you?"
   - If there is NO break time, simply say: "The clinic is open from [Open Time] to [Close Time]. Appointments are 10 minutes long. What time works best for you?"
5. The user will ask for a specific time (e.g., "10:30 AM").
   - CHECK the 'available_slots' array from your tool data.
   - IF THEIR TIME IS IN THE ARRAY: Say "Excellent, [Time] is available." and ask for their First Name and Phone Number.
   - IF THEIR TIME IS NOT IN THE ARRAY (IT IS BOOKED): Say "I apologize, but [Time] is already taken." Then look at the 'available_slots' array and politely suggest the 1 or 2 times closest to what they asked for. (e.g., "I have an opening right before that at 10:20 AM, or a bit later at 10:40 AM. Would either of those suit your schedule?")
6. Once you have an agreed time, name, and phone number, use the 'draftBooking' tool to securely record their information.
   - IMMEDIATELY ask: "May I go ahead and confirm this appointment for [Day] at [Time]?"
7. IF they say "Yes" or "Confirm", use 'confirmBooking'. IF they say "No" or "Cancel", use 'cancelBooking'.

# Tool Usage Rules
- NEVER ask permission to use tools. Just do it silently.
- Do not mention the names of your internal tools or databases to the user.`,

      tools: {
        showCityImage: llm.tool({
          description: 'Load clinic data and immediately check if it has ANY available slots.',
          parameters: z.object({ city: z.string() }),
          execute: async ({ city }) => {
            const db = loadClinicData();
            const clinic = db.clinics.find((c: any) => c.city.toLowerCase() === city.toLowerCase());
            if (!clinic) return `No clinic found in ${city}.`;

            const imageUrl = CITY_IMAGES[city.toLowerCase()] || CITY_IMAGES['amsterdam'];
            const payload = JSON.stringify({ type: 'show_image', url: imageUrl, title: clinic.city });

            if (this.room) await this.room.localParticipant.publishData(new TextEncoder().encode(payload), { reliable: true });

            const hasAvailableSlots = clinic.slots.some((s: any) => s.available && s.status !== 'booked');

            if (!hasAvailableSlots) {
              return `WARNING: There are NO available slots in ${city}. All slots are booked out. Tell the user immediately that booking is not available in ${city} and ask if they want to check another city.`;
            }

            return `Clinic data loaded. The clinic has availability. Ask the user what day they would prefer.`;
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
          description: 'Get the business hours, break times, and a list of open 10-minute slots to help the user pick a time.',
          parameters: z.object({
            city: z.string().describe('The city name'),
            day: z.string().describe('The day of the week (e.g., MONDAY). DO NOT PASS "TODAY".'),
          }),
          execute: async ({ city, day }) => {
            const db = loadClinicData();
            const clinic = db.clinics.find((c: any) => c.city.toLowerCase() === city.toLowerCase());
            if (!clinic) return `No slots found in ${city}.`;

            const targetDay = day.toUpperCase();
            const shift = clinic.slots.find((s: any) => s.day === targetDay && s.available && s.status !== 'booked');

            if (!shift) {
              return `I'm sorry, we don't have any available slots on ${targetDay} in ${city}. Please ask them for another day.`;
            }

            // Generate exact available times, automatically skipping breaks and booked slots
            const availableTimes = get10MinSlots(shift);

            if (availableTimes.length === 0) {
              return `I'm sorry, all slots for ${targetDay} are currently booked out.`;
            }

            // Check if a valid break actually exists
            const bStart = normalizeTime(shift.breakStart);
            const bEnd = normalizeTime(shift.breakEnd);
            const hasBreak = bStart !== null && bEnd !== null;

            // Build the schedule data dynamically
            const scheduleData: any = {
              business_hours: { open: shift.startTime, close: shift.endTime },
              available_slots: availableTimes
            };

            if (hasBreak) {
              scheduleData.break_time = { start: shift.breakStart, end: shift.breakEnd };
            }

            const instructionText = hasBreak 
              ? "State the business hours and the break time, then ask what time they want." 
              : "State the business hours, then ask what time they want. DO NOT mention a break time because there is no break.";

            return `Here is the schedule data: ${JSON.stringify(scheduleData)}. 
            AI INSTRUCTION: ${instructionText} DO NOT READ THE AVAILABLE SLOTS ARRAY. Only use the array if their requested time is missing, so you can suggest the closest adjacent times. Convert all 24-hour times into friendly AM/PM formats when speaking to the user.`;
          }
        }),

        draftBooking: llm.tool({
          description: 'Draft the appointment data BEFORE saving it to the database. Use this after getting the name and phone number.',
          parameters: z.object({
            city: z.string(),
            day: z.string(),
            time: z.string(),
            patient_name: z.string(),
            phone_number: z.string(),
          }),
          execute: async ({ city, day, time, patient_name, phone_number }) => {
            const db = loadClinicData();
            let targetClinic = db.clinics.find((c: any) => c.city.toLowerCase() === city.toLowerCase());
            if (!targetClinic) return `Booking failed. Invalid city.`;

            const normalizedRequestTime = normalizeTime(time) || time;

            const ticket = {
              status: "PENDING CONFIRMATION",
              patient_name: patient_name, 
              phone_number: phone_number, 
              day: day.toUpperCase(), 
              time: normalizedRequestTime,
              clinic_name: targetClinic.clinic_name, 
              address: targetClinic.streetAddress || targetClinic.address
            };

            const payload = JSON.stringify({ type: 'show_ticket', ticket: ticket });
            
            if (this.room) {
              await this.room.localParticipant.publishData(new TextEncoder().encode(payload), { reliable: true });
            }

            return `Booking prepared. You MUST ask the user exactly: "May I go ahead and confirm this appointment for ${day} at ${normalizedRequestTime}?"`;
          }
        }),

        confirmBooking: llm.tool({
          description: 'Finalize and save the booking ONLY AFTER the user explicitly says YES or CONFIRM to the draft.',
          parameters: z.object({
            city: z.string(),
            day: z.string(),
            time: z.string(),
            patient_name: z.string(),
            phone_number: z.string(),
          }),
          execute: async ({ city, day, time, patient_name, phone_number }) => {
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
            
            // Normalize the requested time so it perfectly matches the database
            const normalizedRequestTime = normalizeTime(time);
            
            if (targetShift.booked_appointments.some((b:any) => normalizeTime(b.time) === normalizedRequestTime)) {
              return `I apologize, but that specific time slot was just taken. Please suggest another nearby time.`;
            }

            // Save clean format to database (e.g., "09:10")
            targetShift.booked_appointments.push({ 
              time: normalizedRequestTime, 
              patient_name: patient_name, 
              phone_number: phone_number 
            });
            saveClinicData(db);

            const ticket = {
              status: "CONFIRMED ✅",
              patient_name: patient_name, 
              phone_number: phone_number, 
              day: day.toUpperCase(), 
              time: normalizedRequestTime,
              clinic_name: targetClinic.clinic_name, 
              address: targetClinic.streetAddress || targetClinic.address
            };

            const payload = JSON.stringify({ type: 'show_ticket', ticket: ticket });
            const closeImgPayload = JSON.stringify({ type: 'close_image' });

            if (this.room) {
              await this.room.localParticipant.publishData(new TextEncoder().encode(closeImgPayload), { reliable: true });
              await this.room.localParticipant.publishData(new TextEncoder().encode(payload), { reliable: true });
            }

            return `Successfully saved to database. Tell the user their appointment is fully confirmed and thank them for choosing Health 4 Travel.`;
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
            const normalizedRequestTime = normalizeTime(time);

            for (const c of db.clinics) {
              if (c.city.toLowerCase() === city.toLowerCase()) {
                for (const s of c.slots) {
                  if (s.day === day.toUpperCase() && s.booked_appointments) {
                    const initialLength = s.booked_appointments.length;
                    s.booked_appointments = s.booked_appointments.filter((b:any) => normalizeTime(b.time) !== normalizedRequestTime);
                    if (s.booked_appointments.length < initialLength) {
                      foundAndDeleted = true;
                    }
                  }
                }
              }
            }

            const closeTicketPayload = JSON.stringify({ type: 'show_ticket', ticket: null });
            if (this.room) await this.room.localParticipant.publishData(new TextEncoder().encode(closeTicketPayload), { reliable: true });

            if (foundAndDeleted) {
              saveClinicData(db);
              return `The appointment for ${day} at ${time} has been successfully cancelled from the database. Inform the user gracefully.`;
            } else {
              return `The draft booking has been cancelled. Ask the user if they need help with anything else.`;
            }
          }
        }),
      }
    });

    this.room = room;
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
      
      const llm_model = new inference.LLM({ model: 'openai/gpt-4o-mini' });
      
      const tts = new elevenlabs.TTS({
        apiKey: process.env.ELEVEN_API_KEY!, enableLogging: true, voiceId: process.env.ELEVEN_VOICE_ID!, language: 'en', model: 'eleven_flash_v2_5'
      });

      const session = new voice.AgentSession({
        stt: stt, llm: llm_model, tts: tts, turnDetection: new livekit.turnDetector.MultilingualModel(), vad: ctx.proc.userData.vad! as silero.VAD,
        voiceOptions: { preemptiveGeneration: true, allowInterruptions: true, minInterruptionDuration: 1.2, minInterruptionWords: 5, minEndpointingDelay: 0.6, maxEndpointingDelay: 3.0, maxToolSteps: 10 },
      });

      const agentInstance = new Health4TravelAgent(ctx.room);

      const usageCollector = new metrics.UsageCollector();
      session.on(voice.AgentSessionEventTypes.MetricsCollected, (ev) => { metrics.logMetrics(ev.metrics); usageCollector.collect(ev.metrics); });
      ctx.addShutdownCallback(async () => { console.log(`Usage: ${JSON.stringify(usageCollector.getSummary())}`); });

      await session.start({ agent: agentInstance, room: ctx.room, inputOptions: { noiseCancellation: BackgroundVoiceCancellation() }});

      console.log('✅ AI Session Started successfully');
      await new Promise((resolve) => setTimeout(resolve, 3000));

      await session.say(
        `Hello! Welcome to Health 4 Travel. I am ${agentInstance.agentName}, and I will help you book an appointment with doctor. Which city are you looking to book a doctor in today?`
      );

    } catch (error) {
      console.error('❌ ERROR IN AGENT ENTRY:', error);
    }
  },
});

cli.runApp(new ServerOptions({ 
  agent: fileURLToPath(import.meta.url),
  port: process.env.PORT ? parseInt(process.env.PORT) : 7860 
}));